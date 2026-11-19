/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <calibration/CameraCalibration.h>

#include <gtest/gtest.h>

using namespace calibration;
using MT = CameraProjection::ModelType;

namespace {

CameraProjection makeModel(MT type) {
  Eigen::VectorXd p;
  switch (type) {
    case MT::Linear:
    case MT::Spherical:
      p.resize(4);
      p << 240.0, 241.0, 255.5, 254.0;
      break;
    case MT::KannalaBrandtK3:
      p.resize(8);
      p << 190.97, 190.98, 254.93, 256.90, 0.0034823, 0.00071503, -0.0020532, 0.00020293;
      break;
    case MT::Fisheye624:
      p.resize(15);
      p << 241.1, 318.2, 238.9, -0.025, 0.098, -0.067, 0.0059, 0.0012, -0.0003, 0.0004, -0.0002,
          0.0007, -0.0001, -0.0003, 0.0001;
      break;
    case MT::Fisheye62:
      p.resize(11);
      p << 241.1, 318.2, 238.9, -0.025, 0.098, -0.067, 0.0059, 0.0012, -0.0003, 0.0004, -0.0002;
      break;
  }
  return CameraProjection(type, p);
}

const std::vector<MT> kAllModels =
    {MT::Linear, MT::Spherical, MT::KannalaBrandtK3, MT::Fisheye624, MT::Fisheye62};

std::vector<Eigen::Vector3d> testPoints() {
  std::vector<Eigen::Vector3d> pts;
  for (double x : {-0.6, -0.1, 0.0, 0.3, 0.7}) {
    for (double y : {-0.5, 0.0, 0.2, 0.6}) {
      pts.emplace_back(x, y, 1.0);
    }
  }
  return pts;
}

} // namespace

TEST(CameraProjection, NumParameters) {
  EXPECT_EQ(makeModel(MT::Linear).numParameters(), 4);
  EXPECT_EQ(makeModel(MT::KannalaBrandtK3).numParameters(), 8);
  EXPECT_EQ(makeModel(MT::KannalaBrandtK3).numDistortionParameters(), 4);
  EXPECT_EQ(makeModel(MT::Fisheye624).numParameters(), 15);
  EXPECT_EQ(makeModel(MT::Fisheye624).numProjectionParameters(), 3);
  EXPECT_EQ(makeModel(MT::Fisheye62).numParameters(), 11);
  EXPECT_THROW(CameraProjection(MT::KannalaBrandtK3, Eigen::VectorXd::Zero(4)), std::runtime_error);
}

TEST(CameraProjection, ModelNames) {
  for (MT t : kAllModels) {
    EXPECT_EQ(modelTypeFromString(modelTypeToString(t)), t);
  }
  EXPECT_THROW(modelTypeFromString("Foo"), std::runtime_error);
}

TEST(CameraProjection, ProjectUnprojectRoundTrip) {
  for (MT t : kAllModels) {
    const auto model = makeModel(t);
    for (const auto& pt : testPoints()) {
      const Eigen::Vector2d px = model.project(pt);
      const Eigen::Vector3d ray = model.unproject(px);
      EXPECT_LT((ray / ray.z() - pt / pt.z()).norm(), 1e-6) << modelTypeToString(t);
      EXPECT_LT((model.project(ray) - px).norm(), 1e-6) << modelTypeToString(t);
    }
  }
}

TEST(CameraProjection, JacobiansMatchNumeric) {
  constexpr double kEps = 1e-6;
  for (MT t : kAllModels) {
    const auto model = makeModel(t);
    if (t == MT::Spherical) { // Jacobians are not implemented for this model
      Eigen::Matrix<double, 2, 3> J_pt;
      EXPECT_THROW(model.project(Eigen::Vector3d(0.1, 0.2, 1.0), J_pt), std::runtime_error);
      continue;
    }
    const int nParams = model.numParameters();
    for (const auto& pt : testPoints()) {
      Eigen::Matrix<double, 2, 3> J_pt;
      Eigen::Matrix<double, 2, Eigen::Dynamic> J_par(2, nParams);
      const Eigen::Vector2d px = model.project(pt, J_pt, J_par);

      for (int i = 0; i < 3; i++) {
        Eigen::Vector3d ptP = pt;
        ptP[i] += kEps;
        const Eigen::Vector2d num = (model.project(ptP) - px) / kEps;
        EXPECT_LT((num - J_pt.col(i)).norm(), 1e-3) << modelTypeToString(t) << " pt " << i;
      }
      for (int i = 0; i < nParams; i++) {
        auto modelP = model;
        modelP.projectionParamsMut()[i] += kEps;
        const Eigen::Vector2d num = (modelP.project(pt) - px) / kEps;
        EXPECT_LT((num - J_par.col(i)).norm(), 1e-3) << modelTypeToString(t) << " param " << i;
      }
    }
  }
}

TEST(CameraCalibration, RescaleAndVisibility) {
  const auto proj = makeModel(MT::KannalaBrandtK3);
  CameraCalibration calib(
      "cam0",
      proj.modelName(),
      proj.projectionParams(),
      Sophus::SE3d(),
      512,
      512,
      250.0,
      1.4,
      "serial",
      0.001,
      0.005);
  EXPECT_TRUE(calib.isVisible({255.0, 255.0}));
  EXPECT_FALSE(calib.isVisible({-1.0, 255.0}));
  EXPECT_FALSE(calib.isVisible({2.0, 2.0})); // outside valid radius
  EXPECT_FALSE(calib.project({0.0, 0.0, -1.0}).has_value());
  EXPECT_TRUE(calib.project({0.1, 0.0, 1.0}).has_value());

  const auto half = calib.rescale({256, 256}, 0.5);
  EXPECT_EQ(half.getImageSize(), Eigen::Vector2i(256, 256));
  EXPECT_NEAR(*half.getValidRadius(), 125.0, 1e-12);
  EXPECT_NEAR(half.getFocalLengths().x(), 0.5 * calib.getFocalLengths().x(), 1e-12);
  EXPECT_EQ(half.getReadOutTimeSec(), calib.getReadOutTimeSec());
  const Eigen::Vector3d pt(0.2, -0.1, 1.0);
  // pixel centers: u_half + 0.5 = 0.5 * (u + 0.5)
  const Eigen::Vector2d expected =
      0.5 * (calib.projectNoChecks(pt) + Eigen::Vector2d::Constant(0.5)) -
      Eigen::Vector2d::Constant(0.5);
  EXPECT_LT((half.projectNoChecks(pt) - expected).norm(), 1e-9);
}
