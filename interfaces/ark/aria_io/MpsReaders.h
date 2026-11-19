/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Readers for the MPS (Machine Perception Services) outputs used as VIBA inputs: online
// calibration (.jsonl) and open loop trajectory (.csv). Reimplements the corresponding
// projectaria_tools readers, with no dependency on it.

#pragma once

#include <calibration/CameraCalibration.h>
#include <calibration/ImuCalibration.h>

#include <nlohmann/json.hpp>
#include <sophus/se3.hpp>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace aria_io {

struct OnlineCalibration {
  int64_t trackingTimestampUs = 0;
  int64_t utcTimestampNs = 0;
  std::vector<calibration::CameraCalibration> cameraCalibs;
  std::vector<calibration::ImuCalibration> imuCalibs;
};

/// One json object per line. Supports the format since MPS 1.1 (with optional "ImageSizes" and
/// "ReadoutTimesSec") and the older format where calibrations were stored as strings.
std::vector<OnlineCalibration> readOnlineCalibrations(const std::filesystem::path& path);

/// Inverse of readOnlineCalibrations (single line, current format).
nlohmann::json onlineCalibrationToJson(const OnlineCalibration& calib);

struct OpenLoopTrajectoryPose {
  int64_t trackingTimestampUs = 0;
  int64_t utcTimestampNs = 0;
  std::string sessionUid;
  Sophus::SE3d T_odometry_device;
  Eigen::Vector3d deviceLinearVelocity_odometry;
  Eigen::Vector3d angularVelocity_device;
  Eigen::Vector3d gravity_odometry;
  float qualityScore = 0;
};

/// Throws on error.
std::vector<OpenLoopTrajectoryPose> readOpenLoopTrajectory(const std::filesystem::path& path);

} // namespace aria_io
