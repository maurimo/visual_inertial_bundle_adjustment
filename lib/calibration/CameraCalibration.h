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

#include <calibration/CameraProjection.h>

#include <sophus/se3.hpp>
#include <optional>
#include <string>

namespace calibration {

/// Intrinsics + extrinsics + timing of a camera.
class CameraCalibration {
 public:
  CameraCalibration() = default;
  CameraCalibration(
      const std::string& label,
      CameraProjection::ModelType projectionModelType,
      const Eigen::VectorXd& projectionParams,
      const Sophus::SE3d& T_Device_Camera,
      int imageWidth,
      int imageHeight,
      std::optional<double> maybeValidRadius,
      double maxSolidAngle,
      const std::string& serialNumber = "",
      double timeOffsetSecDeviceCamera = 0.0,
      std::optional<double> maybeReadOutTimeSec = std::nullopt);

  const std::string& getLabel() const {
    return label_;
  }
  const std::string& getSerialNumber() const {
    return serialNumber_;
  }
  const Sophus::SE3d& getT_Device_Camera() const {
    return T_Device_Camera_;
  }
  Sophus::SE3d& getT_Device_CameraMut() {
    return T_Device_Camera_;
  }
  Eigen::Vector2i getImageSize() const {
    return {imageWidth_, imageHeight_};
  }
  double getMaxSolidAngle() const {
    return maxSolidAngle_;
  }
  std::optional<double> getValidRadius() const {
    return maybeValidRadius_;
  }
  double getTimeOffsetSecDeviceCamera() const {
    return timeOffsetSecDeviceCamera_;
  }
  double& getTimeOffsetSecDeviceCameraMut() {
    return timeOffsetSecDeviceCamera_;
  }
  std::optional<double> getReadOutTimeSec() const {
    return maybeReadOutTimeSec_;
  }
  std::optional<double>& getReadOutTimeSecMut() {
    return maybeReadOutTimeSec_;
  }

  const CameraProjection& projection() const {
    return projectionModel_;
  }
  CameraProjection::ModelType modelName() const {
    return projectionModel_.modelName();
  }
  Eigen::Vector2d getPrincipalPoint() const {
    return projectionModel_.getPrincipalPoint();
  }
  Eigen::Vector2d getFocalLengths() const {
    return projectionModel_.getFocalLengths();
  }
  const Eigen::VectorXd& projectionParams() const {
    return projectionModel_.projectionParams();
  }
  Eigen::VectorXd& projectionParamsMut() {
    return projectionModel_.projectionParamsMut();
  }
  int numParameters() const {
    return projectionModel_.numParameters();
  }
  int numProjectionParameters() const {
    return projectionModel_.numProjectionParameters();
  }
  int numDistortionParameters() const {
    return projectionModel_.numDistortionParameters();
  }

  /// true if pixel is in image bounds and inside the valid radius (if any)
  bool isVisible(const Eigen::Vector2d& cameraPixel) const;

  Eigen::Vector2d projectNoChecks(
      const Eigen::Vector3d& pointInCamera,
      Eigen::Ref<Eigen::Matrix<double, 2, 3>> jacobianWrtPoint = NullRef(),
      Eigen::Ref<Eigen::Matrix<double, 2, Eigen::Dynamic>> jacobianWrtParams = NullRef()) const {
    return projectionModel_.project(pointInCamera, jacobianWrtPoint, jacobianWrtParams);
  }

  /// projects, returning nullopt if outside the max solid angle or not visible
  std::optional<Eigen::Vector2d> project(
      const Eigen::Vector3d& pointInCamera,
      Eigen::Ref<Eigen::Matrix<double, 2, 3>> jacobianWrtPoint = NullRef(),
      Eigen::Ref<Eigen::Matrix<double, 2, Eigen::Dynamic>> jacobianWrtParams = NullRef()) const;

  Eigen::Vector3d unprojectNoChecks(const Eigen::Vector2d& cameraPixel) const {
    return projectionModel_.unproject(cameraPixel);
  }

  /// returns a calibration for a rescaled (and optionally cropped) image
  CameraCalibration rescale(
      const Eigen::Vector2i& newResolution,
      double scale,
      const Eigen::Vector2d& originOffset = {0.0, 0.0}) const;

 private:
  std::string label_;
  CameraProjection projectionModel_;
  Sophus::SE3d T_Device_Camera_;
  int imageWidth_ = 0;
  int imageHeight_ = 0;
  std::optional<double> maybeValidRadius_;
  double maxSolidAngle_ = 0.0;
  std::string serialNumber_;
  double timeOffsetSecDeviceCamera_ = 0.0;
  std::optional<double> maybeReadOutTimeSec_;
};

} // namespace calibration
