/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <aria_io/MpsReaders.h>

#include <aria_io/AriaCalibrationJson.h>
#include <csv/CsvReader.h>

#include <fmt/format.h>
#include <fstream>
#include <stdexcept>

namespace aria_io {

namespace {

void replaceAll(std::string& str, const std::string& from, const std::string& to) {
  for (size_t pos = 0; (pos = str.find(from, pos)) != std::string::npos; pos += to.size()) {
    str.replace(pos, from.size(), to);
  }
}

// before Jun 2023 sensor calibrations were stored as python-ish strings
nlohmann::json parseLegacyString(std::string s) {
  replaceAll(s, "True", "true");
  replaceAll(s, "False", "false");
  replaceAll(s, "'", "\"");
  return nlohmann::json::parse(s);
}

int64_t toInt64(const nlohmann::json& j) {
  return j.is_string() ? std::stoll(j.get<std::string>()) : j.get<int64_t>();
}

OnlineCalibration parseOnlineCalibration(const nlohmann::json& doc) {
  OnlineCalibration calib;
  calib.trackingTimestampUs = toInt64(doc.at("tracking_timestamp_us"));
  calib.utcTimestampNs = toInt64(doc.at("utc_timestamp_ns"));

  const bool legacy = doc.at("CameraCalibrations").is_string();
  const nlohmann::json camDoc =
      legacy ? parseLegacyString(doc["CameraCalibrations"]) : doc["CameraCalibrations"];
  const nlohmann::json imuDoc =
      legacy ? parseLegacyString(doc["ImuCalibrations"]) : doc["ImuCalibrations"];

  // device version is determined by number of cameras
  const size_t numCameras = camDoc.size();
  const auto defaults =
      defaultCameraConfigs((numCameras == 4 || numCameras == 5) ? "Oatmeal" : "Aria");

  for (size_t cameraIdx = 0; cameraIdx < numCameras; ++cameraIdx) {
    auto camCalib = parseCameraCalibration(camDoc[cameraIdx], defaults);
    // since MPS 1.1 image sizes and readout times are stored with the calibration
    if (!legacy && doc.contains("ImageSizes")) {
      const auto& imageSize = doc["ImageSizes"].at(cameraIdx);
      std::optional<double> readoutTimeSec;
      if (doc.contains("ReadoutTimesSec")) {
        for (const auto& info : doc["ReadoutTimesSec"]) { // [cameraIdx, readoutTimeSec]
          if (info[0].get<size_t>() == cameraIdx) {
            readoutTimeSec = info[1].get<double>();
            break;
          }
        }
      }
      camCalib = calibration::CameraCalibration(
          camCalib.getLabel(),
          camCalib.modelName(),
          camCalib.projectionParams(),
          camCalib.getT_Device_Camera(),
          imageSize[0].get<int>(),
          imageSize[1].get<int>(),
          camCalib.getValidRadius(),
          camCalib.getMaxSolidAngle(),
          camCalib.getSerialNumber(),
          camCalib.getTimeOffsetSecDeviceCamera(),
          readoutTimeSec);
    }
    calib.cameraCalibs.push_back(std::move(camCalib));
  }
  for (const auto& imuJson : imuDoc) {
    calib.imuCalibs.push_back(parseImuCalibration(imuJson));
  }
  return calib;
}

} // namespace

std::vector<OnlineCalibration> readOnlineCalibrations(const std::filesystem::path& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    throw std::runtime_error(fmt::format("File not found: {}", path.string()));
  }
  std::vector<OnlineCalibration> calibs;
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.find_first_not_of(" \t\r\n") == std::string::npos) {
      continue;
    }
    calibs.push_back(parseOnlineCalibration(nlohmann::json::parse(line)));
  }
  return calibs;
}

nlohmann::json onlineCalibrationToJson(const OnlineCalibration& calib) {
  nlohmann::json json;
  json["tracking_timestamp_us"] = calib.trackingTimestampUs;
  json["utc_timestamp_ns"] = calib.utcTimestampNs;
  nlohmann::json jCameras = nlohmann::json::array();
  nlohmann::json jImageSizes = nlohmann::json::array();
  nlohmann::json jReadoutTimes = nlohmann::json::array();
  for (size_t i = 0; i < calib.cameraCalibs.size(); i++) {
    const auto& cam = calib.cameraCalibs[i];
    jCameras.push_back(cameraCalibrationToJson(cam));
    jImageSizes.push_back({cam.getImageSize().x(), cam.getImageSize().y()});
    if (cam.getReadOutTimeSec().has_value()) {
      jReadoutTimes.push_back({i, *cam.getReadOutTimeSec()});
    }
  }
  json["CameraCalibrations"] = jCameras;
  json["ImageSizes"] = jImageSizes;
  if (!jReadoutTimes.empty()) {
    json["ReadoutTimesSec"] = jReadoutTimes;
  }
  nlohmann::json jImus = nlohmann::json::array();
  for (const auto& imu : calib.imuCalibs) {
    jImus.push_back(imuCalibrationToJson(imu));
  }
  json["ImuCalibrations"] = jImus;
  return json;
}

std::vector<OpenLoopTrajectoryPose> readOpenLoopTrajectory(const std::filesystem::path& path) {
  csv::Reader<20> reader(
      path,
      {"tracking_timestamp_us",
       "utc_timestamp_ns",
       "session_uid",
       "tx_odometry_device",
       "ty_odometry_device",
       "tz_odometry_device",
       "qx_odometry_device",
       "qy_odometry_device",
       "qz_odometry_device",
       "qw_odometry_device",
       "device_linear_velocity_x_odometry",
       "device_linear_velocity_y_odometry",
       "device_linear_velocity_z_odometry",
       "angular_velocity_x_device",
       "angular_velocity_y_device",
       "angular_velocity_z_device",
       "gravity_x_odometry",
       "gravity_y_odometry",
       "gravity_z_odometry",
       "quality_score"});

  std::vector<OpenLoopTrajectoryPose> trajectory;
  OpenLoopTrajectoryPose p;
  Eigen::Vector3d t;
  Eigen::Quaterniond q;
  while (reader.readRow(
      p.trackingTimestampUs,
      p.utcTimestampNs,
      p.sessionUid,
      t.x(),
      t.y(),
      t.z(),
      q.x(),
      q.y(),
      q.z(),
      q.w(),
      p.deviceLinearVelocity_odometry.x(),
      p.deviceLinearVelocity_odometry.y(),
      p.deviceLinearVelocity_odometry.z(),
      p.angularVelocity_device.x(),
      p.angularVelocity_device.y(),
      p.angularVelocity_device.z(),
      p.gravity_odometry.x(),
      p.gravity_odometry.y(),
      p.gravity_odometry.z(),
      p.qualityScore)) {
    p.T_odometry_device = Sophus::SE3d(q, t);
    trajectory.push_back(p);
  }
  return trajectory;
}

} // namespace aria_io
