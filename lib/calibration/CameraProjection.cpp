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

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <type_traits>

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
    case MT::PinholeRadTan:
      return f(PinholeRadTan4{});
    case MT::PinholeRadTanK3:
      return f(PinholeRadTan5{});
    case MT::DoubleSphere:
      return f(DoubleSphereProjection{});
    case MT::ExtendedUnified:
      return f(ExtendedUnifiedProjection{});
    case MT::Unified:
      return f(UnifiedProjection{});
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

const std::vector<CameraProjection::ModelType>& allModelTypes() {
  using MT = CameraProjection::ModelType;
  static const std::vector<MT> kAll = {
      MT::Linear,
      MT::Spherical,
      MT::KannalaBrandtK3,
      MT::Fisheye624,
      MT::Fisheye62,
      MT::PinholeRadTan,
      MT::PinholeRadTanK3,
      MT::DoubleSphere,
      MT::ExtendedUnified,
      MT::Unified};
  return kAll;
}

namespace {
// FisheyeRadTanThinPrism's kName/kDescription are declared but not defined (as upstream), so they
// are not used here
template <class P>
constexpr bool isFisheyeRadTanThinPrism() {
  return std::is_same_v<P, Fisheye624> || std::is_same_v<P, Fisheye62>;
}
} // namespace

std::string modelTypeToString(CameraProjection::ModelType type) {
  using MT = CameraProjection::ModelType;
  switch (type) {
    case MT::Fisheye624:
      return "FisheyeRadTanThinPrism"; // name in Aria calibration files
    case MT::Fisheye62:
      return "Fisheye62";
    default:
      return visitModel(type, [](auto proj) -> std::string {
        using P = decltype(proj);
        if constexpr (isFisheyeRadTanThinPrism<P>()) {
          return "";
        } else {
          return P::kName;
        }
      });
  }
}

std::string modelTypeDescription(CameraProjection::ModelType type) {
  using MT = CameraProjection::ModelType;
  switch (type) {
    case MT::Fisheye624:
      return "fx, cx, cy, k_0, k_1, k_2, k_3, k_4, k_5, p_0, p_1, s_0, s_1, s_2, s_3";
    case MT::Fisheye62:
      return "fx, cx, cy, k_0, k_1, k_2, k_3, k_4, k_5, p_0, p_1";
    default:
      return visitModel(type, [](auto proj) -> std::string {
        using P = decltype(proj);
        if constexpr (isFisheyeRadTanThinPrism<P>()) {
          return "";
        } else {
          return P::kDescription;
        }
      });
  }
}

CameraProjection::ModelType modelTypeFromString(const std::string& name) {
  using MT = CameraProjection::ModelType;
  auto lower = [](std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
  };
  const std::string key = lower(name);
  for (MT t : allModelTypes()) {
    if (lower(modelTypeToString(t)) == key) {
      return t;
    }
  }
  static const std::vector<std::pair<std::string, MT>> kAliases = {
      {"fisheye624", MT::Fisheye624},
      {"pinhole", MT::Linear},
      {"kannalabrandt", MT::KannalaBrandtK3},
      {"kannalabrandt4", MT::KannalaBrandtK3},
      {"kb4", MT::KannalaBrandtK3},
      {"equidistant", MT::KannalaBrandtK3},
      {"radtan", MT::PinholeRadTan},
      {"pinholeradtan4", MT::PinholeRadTan},
      {"pinholeradtan5", MT::PinholeRadTanK3},
      {"ds", MT::DoubleSphere},
      {"eucm", MT::ExtendedUnified},
      {"ucm", MT::Unified},
  };
  for (const auto& [alias, t] : kAliases) {
    if (alias == key) {
      return t;
    }
  }
  throw std::runtime_error("Unknown camera model name: '" + name + "'");
}

} // namespace calibration
