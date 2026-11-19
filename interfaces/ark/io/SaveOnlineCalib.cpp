/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <io/SaveOnlineCalib.h>

#include <aria_io/MpsReaders.h>
#include <nlohmann/json.hpp>
#include <session_data/ImuCalibConversion.h>
#include <viba/common/Enumerate.h>
#include <fstream>

#define DEFAULT_LOG_CHANNEL "ViBa::SaveOnlineCalib"
#include <logging/Log.h>

namespace visual_inertial_ba {

void saveOnlineCalib(
    const SingleSessionProblem& prob,
    const SessionData& fData,
    const Matcher& matcher,
    const std::filesystem::path& outputPath) {
  const int numCameras = fData.slamInfo.cameraSerialNumbers.size();
  const int numImus = fData.slamInfo.imuLabels.size();
  auto rigIndices = prob.sortedRigIndices();

  std::ofstream ofs(outputPath);
  for (int64_t rigIndex : rigIndices) {
    const auto& inputRig = fData.inertialPoses.at(matcher.rigIndexToEvolvingStateIndex.at(rigIndex));
    aria_io::OnlineCalibration calib;
    calib.trackingTimestampUs = inputRig.timestamp_us;
    calib.utcTimestampNs = inputRig.utc_timestamp_ns;

    for (int s = 0; s < numCameras; s++) { // set camera intrinsics
      calibration::CameraCalibration model = prob.cameraModel(rigIndex, s).var.value.model;
      SE3 T_Cam_BodyImu = prob.T_Cam_BodyImu(rigIndex, s).var.value;
      model.getT_Device_CameraMut() =
          fData.slamInfo.T_bodyImu_device.inverse() * T_Cam_BodyImu.inverse();
      calib.cameraCalibs.push_back(std::move(model));
    }

    for (int s = 0; s < numImus; s++) {
      const auto& imuModel = prob.imuCalib(rigIndex, s).var.value.modelParams;
      SE3 T_Imu_BodyImu = s == 0 ? SE3() : prob.T_Imu_BodyImu(rigIndex, s).var.value;
      calib.imuCalibs.push_back(toImuCalibration(
          imuModel,
          fData.slamInfo.imuLabels[s],
          fData.slamInfo.T_bodyImu_device.inverse() * T_Imu_BodyImu.inverse()));
    }

    const nlohmann::json json = aria_io::onlineCalibrationToJson(calib);
    ofs << json.dump() << std::endl;
  }

  XR_LOGI("Online calibration ({} entries) saved to {}", rigIndices.size(), outputPath.string());
}

} // namespace visual_inertial_ba
