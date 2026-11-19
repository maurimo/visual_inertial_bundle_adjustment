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

#include <calibration/NullRef.h>
#include <calibration/projections/FisheyeRadTanThinPrism.h>
#include <calibration/projections/KannalaBrandtK3.h>
#include <calibration/projections/Linear.h>
#include <calibration/projections/Spherical.h>

#include <Eigen/Core>
#include <string>
#include <variant>

namespace calibration {

/// A camera projection model (type + parameter vector), see the headers in projections/ for the
/// definition of each model.
class CameraProjection {
 public:
  enum class ModelType {
    Linear, ///< Linear pinhole projection: fx, fy, cx, cy
    Spherical, ///< Spherical projection, linear in angular space: fx, fy, cx, cy
    KannalaBrandtK3, ///< Kannala-Brandt (equidistant fisheye): fx, fy, cx, cy, k0..k3
    Fisheye624, ///< f, cx, cy, k0..k5, p0, p1, s0..s3
    Fisheye62, ///< f, cx, cy, k0..k5, p0, p1
  };

  CameraProjection() = default;
  CameraProjection(ModelType type, const Eigen::VectorXd& projectionParams);

  ModelType modelName() const {
    return modelName_;
  }
  const Eigen::VectorXd& projectionParams() const {
    return projectionParams_;
  }
  Eigen::VectorXd& projectionParamsMut() {
    return projectionParams_;
  }

  int numParameters() const;
  int numProjectionParameters() const; // focal(s) and principal point
  int numDistortionParameters() const;

  Eigen::Vector2d project(
      const Eigen::Vector3d& pointInCamera,
      Eigen::Ref<Eigen::Matrix<double, 2, 3>> jacobianWrtPoint = NullRef(),
      Eigen::Ref<Eigen::Matrix<double, 2, Eigen::Dynamic>> jacobianWrtParams = NullRef()) const;

  /// Unprojects to a 3d ray (not normalized). No checks performed.
  Eigen::Vector3d unproject(const Eigen::Vector2d& cameraPixel) const;

  Eigen::Vector2d getPrincipalPoint() const;
  Eigen::Vector2d getFocalLengths() const;

  /// scales the projection parameters as the image scales
  void scaleParams(double scale);
  /// translates the origin by (offsetU, offsetV)
  void subtractFromOrigin(double offsetU, double offsetV);

 private:
  using ProjectionVariant = std::variant<
      LinearProjection,
      SphericalProjection,
      KannalaBrandtK3Projection,
      Fisheye624,
      Fisheye62>;

  ModelType modelName_ = ModelType::Linear;
  Eigen::VectorXd projectionParams_;
  ProjectionVariant projectionVariant_;
};

/// Name used in calibration files ("FisheyeRadTanThinPrism", "KannalaBrandtK3", ...).
std::string modelTypeToString(CameraProjection::ModelType type);
/// Throws on unknown names.
CameraProjection::ModelType modelTypeFromString(const std::string& name);

} // namespace calibration
