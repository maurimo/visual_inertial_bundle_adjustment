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

#include <calibration/CameraCalibration.h>

#include <Eigen/Geometry>
#include <cmath>

namespace calibration {

CameraCalibration::CameraCalibration(
    const std::string& label,
    CameraProjection::ModelType projectionModelType,
    const Eigen::VectorXd& projectionParams,
    const Sophus::SE3d& T_Device_Camera,
    int imageWidth,
    int imageHeight,
    std::optional<double> maybeValidRadius,
    double maxSolidAngle,
    const std::string& serialNumber,
    double timeOffsetSecDeviceCamera,
    std::optional<double> maybeReadOutTimeSec)
    : label_(label),
      projectionModel_(projectionModelType, projectionParams),
      T_Device_Camera_(T_Device_Camera),
      imageWidth_(imageWidth),
      imageHeight_(imageHeight),
      maybeValidRadius_(maybeValidRadius),
      maxSolidAngle_(maxSolidAngle),
      serialNumber_(serialNumber),
      timeOffsetSecDeviceCamera_(timeOffsetSecDeviceCamera),
      maybeReadOutTimeSec_(maybeReadOutTimeSec) {}

bool CameraCalibration::isVisible(const Eigen::Vector2d& cameraPixel) const {
  // pixel (x, y) in discrete coord space has its center at (x, y) in continuous space, so the
  // valid image area is [-0.5, width - 0.5] x [-0.5, height - 0.5]
  const Eigen::AlignedBox2d imgBound(
      Eigen::Vector2d(-0.5, -0.5), Eigen::Vector2d(imageWidth_ - 0.5, imageHeight_ - 0.5));
  if (!imgBound.contains(cameraPixel)) {
    return false;
  }
  if (!maybeValidRadius_) {
    return true;
  }
  return (cameraPixel - getPrincipalPoint()).squaredNorm() <=
      *maybeValidRadius_ * *maybeValidRadius_;
}

std::optional<Eigen::Vector2d> CameraCalibration::project(
    const Eigen::Vector3d& pointInCamera,
    Eigen::Ref<Eigen::Matrix<double, 2, 3>> jacobianWrtPoint,
    Eigen::Ref<Eigen::Matrix<double, 2, Eigen::Dynamic>> jacobianWrtParams) const {
  const double solidAngle = std::atan2(pointInCamera.head<2>().norm(), pointInCamera.z());
  if (solidAngle <= maxSolidAngle_) {
    const Eigen::Vector2d cameraPixel =
        projectNoChecks(pointInCamera, jacobianWrtPoint, jacobianWrtParams);
    if (isVisible(cameraPixel)) {
      return cameraPixel;
    }
  }
  return std::nullopt;
}

CameraCalibration CameraCalibration::rescale(
    const Eigen::Vector2i& newResolution,
    double scale,
    const Eigen::Vector2d& originOffset) const {
  CameraCalibration camCalib(*this);
  camCalib.projectionModel_.subtractFromOrigin(originOffset.x(), originOffset.y());
  camCalib.projectionModel_.scaleParams(scale);
  if (camCalib.maybeValidRadius_) {
    *camCalib.maybeValidRadius_ *= scale;
  }
  camCalib.imageWidth_ = newResolution.x();
  camCalib.imageHeight_ = newResolution.y();
  return camCalib;
}

} // namespace calibration
