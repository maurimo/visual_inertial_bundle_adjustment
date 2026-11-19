/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * Derived from projectaria_tools (core/calibration), reduced to what Visual Inertial Bundle
 * Adjustment needs.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <sophus/se3.hpp>
#include <Eigen/Core>
#include <string>

namespace calibration {

/// Linear sensor model: raw = rectificationMatrix * rectified + bias
struct LinearRectificationModel3d {
  Eigen::Matrix3d rectificationMatrix = Eigen::Matrix3d::Identity();
  Eigen::Vector3d bias = Eigen::Vector3d::Zero();

  const Eigen::Matrix3d& getRectification() const {
    return rectificationMatrix;
  }
  const Eigen::Vector3d& getBias() const {
    return bias;
  }
  Eigen::Vector3d rawToRectified(const Eigen::Vector3d& raw) const {
    return rectificationMatrix.inverse() * (raw - bias);
  }
  Eigen::Vector3d rectifiedToRaw(const Eigen::Vector3d& rectified) const {
    return rectificationMatrix * rectified + bias;
  }
};

/// IMU intrinsics (accel/gyro linear models), extrinsics and time offsets.
class ImuCalibration {
 public:
  ImuCalibration() = default;
  ImuCalibration(
      const std::string& label,
      const Eigen::Matrix3d& rectificationMatrixAccel,
      const Eigen::Vector3d& biasAccel,
      const Eigen::Matrix3d& rectificationMatrixGyro,
      const Eigen::Vector3d& biasGyro,
      const Sophus::SE3d& T_Device_Imu,
      double timeOffsetSecDeviceAccel = 0.0,
      double timeOffsetSecDeviceGyro = 0.0)
      : label_(label),
        accel_{rectificationMatrixAccel, biasAccel},
        gyro_{rectificationMatrixGyro, biasGyro},
        T_Device_Imu_(T_Device_Imu),
        timeOffsetSecDeviceAccel_(timeOffsetSecDeviceAccel),
        timeOffsetSecDeviceGyro_(timeOffsetSecDeviceGyro) {}

  const std::string& getLabel() const {
    return label_;
  }
  const Sophus::SE3d& getT_Device_Imu() const {
    return T_Device_Imu_;
  }
  const LinearRectificationModel3d& getAccelModel() const {
    return accel_;
  }
  const LinearRectificationModel3d& getGyroModel() const {
    return gyro_;
  }
  double getTimeOffsetSecDeviceAccel() const {
    return timeOffsetSecDeviceAccel_;
  }
  double getTimeOffsetSecDeviceGyro() const {
    return timeOffsetSecDeviceGyro_;
  }

 private:
  std::string label_;
  LinearRectificationModel3d accel_;
  LinearRectificationModel3d gyro_;
  Sophus::SE3d T_Device_Imu_;
  double timeOffsetSecDeviceAccel_ = 0.0;
  double timeOffsetSecDeviceGyro_ = 0.0;
};

} // namespace calibration
