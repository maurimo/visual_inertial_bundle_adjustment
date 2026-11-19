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

#include <calibration/CameraProjection.h>

#include <stdexcept>

namespace calibration {

namespace {
template <typename F>
decltype(auto) visitModel(CameraProjection::ModelType type, F&& f) {
  using MT = CameraProjection::ModelType;
  switch (type) {
    case MT::Linear:
      return f(LinearProjection{});
    case MT::Spherical:
      return f(SphericalProjection{});
    case MT::KannalaBrandtK3:
      return f(KannalaBrandtK3Projection{});
    case MT::Fisheye624:
      return f(Fisheye624{});
    case MT::Fisheye62:
      return f(Fisheye62{});
  }
  throw std::runtime_error("Unrecognized camera model.");
}
} // namespace

CameraProjection::CameraProjection(ModelType type, const Eigen::VectorXd& projectionParams)
    : modelName_(type),
      projectionParams_(projectionParams),
      projectionVariant_(
          visitModel(type, [](auto proj) -> ProjectionVariant { return proj; })) {
  if (projectionParams_.size() != numParameters()) {
    throw std::runtime_error(
        "Camera model " + modelTypeToString(type) + " expects " +
        std::to_string(numParameters()) + " parameters, got " +
        std::to_string(projectionParams_.size()));
  }
}

int CameraProjection::numParameters() const {
  return std::visit(
      [](auto&& p) -> int { return std::decay_t<decltype(p)>::kNumParams; }, projectionVariant_);
}

int CameraProjection::numProjectionParameters() const {
  return std::visit(
      [](auto&& p) -> int {
        using T = std::decay_t<decltype(p)>;
        return T::kNumParams - T::kNumDistortionParams;
      },
      projectionVariant_);
}

int CameraProjection::numDistortionParameters() const {
  return std::visit(
      [](auto&& p) -> int { return std::decay_t<decltype(p)>::kNumDistortionParams; },
      projectionVariant_);
}

Eigen::Vector2d CameraProjection::getFocalLengths() const {
  return std::visit(
      [this](auto&& p) -> Eigen::Vector2d {
        using T = std::decay_t<decltype(p)>;
        return {projectionParams_(T::kFocalXIdx), projectionParams_(T::kFocalYIdx)};
      },
      projectionVariant_);
}

Eigen::Vector2d CameraProjection::getPrincipalPoint() const {
  return std::visit(
      [this](auto&& p) -> Eigen::Vector2d {
        using T = std::decay_t<decltype(p)>;
        return {
            projectionParams_(T::kPrincipalPointColIdx),
            projectionParams_(T::kPrincipalPointRowIdx)};
      },
      projectionVariant_);
}

Eigen::Vector2d CameraProjection::project(
    const Eigen::Vector3d& pointInCamera,
    Eigen::Ref<Eigen::Matrix<double, 2, 3>> jacobianWrtPoint,
    Eigen::Ref<Eigen::Matrix<double, 2, Eigen::Dynamic>> jacobianWrtParams) const {
  return std::visit(
      [&](auto&& p) -> Eigen::Vector2d {
        using T = std::decay_t<decltype(p)>;
        return T::project(
            pointInCamera,
            projectionParams_,
            isNull(jacobianWrtPoint) ? nullptr : &jacobianWrtPoint,
            isNull(jacobianWrtParams) ? nullptr : &jacobianWrtParams);
      },
      projectionVariant_);
}

Eigen::Vector3d CameraProjection::unproject(const Eigen::Vector2d& cameraPixel) const {
  return std::visit(
      [&](auto&& p) -> Eigen::Vector3d {
        using T = std::decay_t<decltype(p)>;
        return T::unproject(cameraPixel, projectionParams_);
      },
      projectionVariant_);
}

void CameraProjection::scaleParams(double scale) {
  std::visit(
      [&](auto&& p) {
        using T = std::decay_t<decltype(p)>;
        T::scaleParams(scale, projectionParams_);
      },
      projectionVariant_);
}

void CameraProjection::subtractFromOrigin(double offsetU, double offsetV) {
  std::visit(
      [&](auto&& p) {
        using T = std::decay_t<decltype(p)>;
        T::subtractFromOrigin(offsetU, offsetV, projectionParams_);
      },
      projectionVariant_);
}

std::string modelTypeToString(CameraProjection::ModelType type) {
  using MT = CameraProjection::ModelType;
  switch (type) {
    case MT::Linear:
      return "Linear";
    case MT::Spherical:
      return "Spherical";
    case MT::KannalaBrandtK3:
      return "KannalaBrandtK3";
    case MT::Fisheye624:
      return "FisheyeRadTanThinPrism";
    case MT::Fisheye62:
      return "Fisheye62";
  }
  return "Unknown";
}

CameraProjection::ModelType modelTypeFromString(const std::string& name) {
  using MT = CameraProjection::ModelType;
  for (MT t : {MT::Linear, MT::Spherical, MT::KannalaBrandtK3, MT::Fisheye624, MT::Fisheye62}) {
    if (modelTypeToString(t) == name) {
      return t;
    }
  }
  if (name == "Fisheye624") {
    return MT::Fisheye624;
  }
  throw std::runtime_error("Unknown camera model name: '" + name + "'");
}

} // namespace calibration
