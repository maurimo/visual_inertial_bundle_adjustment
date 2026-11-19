/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <aria_io/AriaCalibrationJson.h>

#include <fmt/format.h>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace aria_io {

using calibration::CameraCalibration;
using calibration::CameraProjection;
using calibration::ImuCalibration;

namespace {

template <int Size>
Eigen::Matrix<double, Size, 1> vectorFromJson(const nlohmann::json& j) {
  if (!j.is_array() || (Size != Eigen::Dynamic && (int)j.size() != Size)) {
    throw std::runtime_error(fmt::format("expected array of size {}, got: {}", Size, j.dump()));
  }
  Eigen::Matrix<double, Size, 1> v;
  if constexpr (Size == Eigen::Dynamic) {
    v.resize(j.size());
  }
  for (size_t i = 0; i < j.size(); i++) {
    v[i] = j[i].get<double>();
  }
  return v;
}

template <typename Derived>
nlohmann::json vectorToJson(const Eigen::MatrixBase<Derived>& v) {
  nlohmann::json j = nlohmann::json::array();
  for (Eigen::Index i = 0; i < v.size(); i++) {
    j.push_back(v[i]);
  }
  return j;
}

Eigen::Matrix3d matrix3FromJson(const nlohmann::json& j) {
  Eigen::Matrix3d m;
  for (int i = 0; i < 3; i++) {
    m.row(i) = vectorFromJson<3>(j.at(i)).transpose();
  }
  return m;
}

nlohmann::json matrix3ToJson(const Eigen::Matrix3d& m) {
  nlohmann::json j = nlohmann::json::array();
  for (int i = 0; i < 3; i++) {
    j.push_back(vectorToJson(m.row(i).transpose()));
  }
  return j;
}

// {"Translation": [x, y, z], "UnitQuaternion": [w, [x, y, z]]}
Sophus::SE3d se3FromJson(const nlohmann::json& j) {
  const Eigen::Vector3d translation = vectorFromJson<3>(j.at("Translation"));
  const auto& jq = j.at("UnitQuaternion");
  if (jq.size() != 2) {
    throw std::runtime_error("UnitQuaternion must be [w, [x, y, z]]");
  }
  const Eigen::Vector3d qv = vectorFromJson<3>(jq[1]);
  const Eigen::Quaterniond q(jq[0].get<double>(), qv.x(), qv.y(), qv.z());
  return Sophus::SE3d(q.normalized(), translation);
}

nlohmann::json se3ToJson(const Sophus::SE3d& T) {
  nlohmann::json j;
  j["Translation"] = vectorToJson(T.translation());
  j["UnitQuaternion"] = nlohmann::json::array();
  j["UnitQuaternion"].push_back(T.unit_quaternion().w());
  j["UnitQuaternion"].push_back(vectorToJson(T.unit_quaternion().vec()));
  return j;
}

std::string projectionDescription(CameraProjection::ModelType type) {
  return calibration::modelTypeDescription(type);
}

std::pair<Eigen::Matrix3d, Eigen::Vector3d> parseRectModel(const nlohmann::json& j) {
  return {
      matrix3FromJson(j.at("Model").at("RectificationMatrix")),
      vectorFromJson<3>(j.at("Bias").at("Offset"))};
}

std::string lowerCase(std::string s) {
  for (auto& c : s) {
    c = std::tolower(c);
  }
  return s;
}

} // namespace

std::map<std::string, CameraConfigData> defaultCameraConfigs(const std::string& deviceClass) {
  const std::string dc = lowerCase(deviceClass);
  if (dc.rfind("aria", 0) == 0) { // Gen1
    return {
        {"camera-slam-left", {640, 480, 1.4, 330.0}},
        {"camera-slam-right", {640, 480, 1.4, 330.0}},
        {"camera-rgb", {2880, 2880, 1.0, 1415.0}},
        {"camera-et-left", {640, 480, 0.75, std::nullopt}},
        {"camera-et-right", {640, 480, 0.75, std::nullopt}},
    };
  }
  if (dc == "oatmeal") { // Gen2
    return {
        {"slam-front-left", {512, 512, 1.4, 300.0}},
        {"slam-front-right", {512, 512, 1.4, 300.0}},
        {"slam-side-left", {512, 512, 1.4, 300.0}},
        {"slam-side-right", {512, 512, 1.4, 300.0}},
        {"camera-rgb", {4032, 3024, 1.4, 2202.0}},
        {"camera-et-left", {400, 400, 0.75, std::nullopt}},
        {"camera-et-right", {400, 400, 0.75, std::nullopt}},
    };
  }
  return {};
}

CameraCalibration parseCameraCalibration(
    const nlohmann::json& json,
    const std::map<std::string, CameraConfigData>& defaultConfigs) {
  const auto modelType =
      calibration::modelTypeFromString(json.at("Projection").at("Name").get<std::string>());
  const Eigen::VectorXd params = vectorFromJson<Eigen::Dynamic>(json.at("Projection").at("Params"));
  const std::string label = json.at("Label");
  const std::string serialNumber = json.value("SerialNumber", "");
  const Sophus::SE3d T_Device_Camera = se3FromJson(json.at("T_Device_Camera"));
  const double timeOffsetSec = json.value("TimeOffsetSec_Device_Camera", 0.0);

  CameraConfigData config;
  if (json.contains("ConfigData")) {
    const auto& jc = json["ConfigData"];
    config.imageWidth = jc.at("ImageWidth");
    config.imageHeight = jc.at("ImageHeight");
    config.maxSolidAngle = jc.at("MaxSolidAngle");
    if (jc.contains("ValidRadius")) {
      config.maybeValidRadius = jc["ValidRadius"].get<double>();
    }
  } else {
    auto it = defaultConfigs.find(label);
    if (it == defaultConfigs.end()) {
      throw std::runtime_error(
          fmt::format("Camera '{}' has no ConfigData, and no default config is known", label));
    }
    config = it->second;
  }

  return CameraCalibration(
      label,
      modelType,
      params,
      T_Device_Camera,
      config.imageWidth,
      config.imageHeight,
      config.maybeValidRadius,
      config.maxSolidAngle,
      serialNumber,
      timeOffsetSec);
}

ImuCalibration parseImuCalibration(const nlohmann::json& json) {
  const auto [accelMat, accelBias] = parseRectModel(json.at("Accelerometer"));
  const auto [gyroMat, gyroBias] = parseRectModel(json.at("Gyroscope"));
  return ImuCalibration(
      json.at("Label"),
      accelMat,
      accelBias,
      gyroMat,
      gyroBias,
      se3FromJson(json.at("T_Device_Imu")),
      json.at("Accelerometer").value("TimeOffsetSec_Device_Accel", 0.0),
      json.at("Gyroscope").value("TimeOffsetSec_Device_Gyro", 0.0));
}

nlohmann::json cameraCalibrationToJson(const CameraCalibration& camCalib) {
  nlohmann::json j;
  j["Calibrated"] = true;
  j["Label"] = camCalib.getLabel();
  j["SerialNumber"] = camCalib.getSerialNumber();
  j["T_Device_Camera"] = se3ToJson(camCalib.getT_Device_Camera());
  j["Projection"]["Description"] = projectionDescription(camCalib.modelName());
  j["Projection"]["Name"] = calibration::modelTypeToString(camCalib.modelName());
  j["Projection"]["Params"] = vectorToJson(camCalib.projectionParams());
  if (camCalib.getTimeOffsetSecDeviceCamera() != 0.0) {
    j["TimeOffsetSec_Device_Camera"] = camCalib.getTimeOffsetSecDeviceCamera();
  }
  auto& jc = j["ConfigData"];
  jc["ImageWidth"] = camCalib.getImageSize().x();
  jc["ImageHeight"] = camCalib.getImageSize().y();
  jc["MaxSolidAngle"] = camCalib.getMaxSolidAngle();
  if (camCalib.getValidRadius().has_value()) {
    jc["ValidRadius"] = *camCalib.getValidRadius();
  }
  return j;
}

nlohmann::json imuCalibrationToJson(const ImuCalibration& imuCalib) {
  nlohmann::json j;
  j["Label"] = imuCalib.getLabel();
  j["SerialNumber"] = "";
  j["Calibrated"] = true;
  j["T_Device_Imu"] = se3ToJson(imuCalib.getT_Device_Imu());
  auto& ja = j["Accelerometer"];
  ja["Model"]["Name"] = "UpperTriagonalLinear";
  ja["Model"]["RectificationMatrix"] = matrix3ToJson(imuCalib.getAccelModel().getRectification());
  ja["Bias"]["Name"] = "Constant";
  ja["Bias"]["Offset"] = vectorToJson(imuCalib.getAccelModel().getBias());
  ja["TimeOffsetSec_Device_Accel"] = imuCalib.getTimeOffsetSecDeviceAccel();
  auto& jg = j["Gyroscope"];
  jg["Model"]["Name"] = "Linear";
  jg["Model"]["RectificationMatrix"] = matrix3ToJson(imuCalib.getGyroModel().getRectification());
  jg["Bias"]["Name"] = "Constant";
  jg["Bias"]["Offset"] = vectorToJson(imuCalib.getGyroModel().getBias());
  jg["TimeOffsetSec_Device_Gyro"] = imuCalib.getTimeOffsetSecDeviceGyro();
  return j;
}

std::vector<std::string> DeviceCalibration::getCameraLabels() const {
  std::vector<std::string> labels;
  for (const auto& [label, _] : cameraCalibs) {
    labels.push_back(label);
  }
  return labels;
}

std::vector<std::string> DeviceCalibration::getImuLabels() const {
  std::vector<std::string> labels;
  for (const auto& [label, _] : imuCalibs) {
    labels.push_back(label);
  }
  return labels;
}

std::optional<CameraCalibration> DeviceCalibration::getCameraCalib(const std::string& label) const {
  auto it = cameraCalibs.find(label);
  return it == cameraCalibs.end() ? std::nullopt : std::make_optional(it->second);
}

std::optional<ImuCalibration> DeviceCalibration::getImuCalib(const std::string& label) const {
  auto it = imuCalibs.find(label);
  return it == imuCalibs.end() ? std::nullopt : std::make_optional(it->second);
}

std::optional<Sophus::SE3d> DeviceCalibration::getT_Device_Sensor(const std::string& label) const {
  if (auto it = imuCalibs.find(label); it != imuCalibs.end()) {
    return it->second.getT_Device_Imu();
  }
  if (auto it = cameraCalibs.find(label); it != cameraCalibs.end()) {
    return it->second.getT_Device_Camera();
  }
  return std::nullopt;
}

DeviceCalibration deviceCalibrationFromJson(const std::string& jsonStr) {
  const auto json = nlohmann::json::parse(jsonStr);
  DeviceCalibration calib;
  if (json.contains("DeviceClassInfo")) {
    calib.deviceClass = json["DeviceClassInfo"].value("DeviceClass", "");
  }
  const auto defaults = defaultCameraConfigs(calib.deviceClass);
  if (json.contains("CameraCalibrations")) {
    for (const auto& camJson : json["CameraCalibrations"]) {
      auto camCalib = parseCameraCalibration(camJson, defaults);
      calib.cameraCalibs.emplace(camCalib.getLabel(), std::move(camCalib));
    }
  }
  if (json.contains("ImuCalibrations")) {
    for (const auto& imuJson : json["ImuCalibrations"]) {
      auto imuCalib = parseImuCalibration(imuJson);
      calib.imuCalibs.emplace(imuCalib.getLabel(), std::move(imuCalib));
    }
  }
  return calib;
}

DeviceCalibration readDeviceCalibration(const std::filesystem::path& path) {
  std::ifstream ifs(path);
  if (!ifs) {
    throw std::runtime_error(fmt::format("File not found: {}", path.string()));
  }
  std::stringstream buffer;
  buffer << ifs.rdbuf();
  return deviceCalibrationFromJson(buffer.str());
}

std::string deviceCalibrationToJson(const DeviceCalibration& calib) {
  nlohmann::json json;
  json["DeviceClassInfo"]["DeviceClass"] = calib.deviceClass;
  json["CameraCalibrations"] = nlohmann::json::array();
  for (const auto& [label, camCalib] : calib.cameraCalibs) {
    json["CameraCalibrations"].push_back(cameraCalibrationToJson(camCalib));
  }
  json["ImuCalibrations"] = nlohmann::json::array();
  for (const auto& [label, imuCalib] : calib.imuCalibs) {
    json["ImuCalibrations"].push_back(imuCalibrationToJson(imuCalib));
  }
  return json.dump(2);
}

} // namespace aria_io
