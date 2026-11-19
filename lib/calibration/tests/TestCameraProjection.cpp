/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <calibration/CameraCalibration.h>

#include <gtest/gtest.h>

#include <algorithm>

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
    case MT::PinholeRadTan: // EuRoC cam0
      p.resize(8);
      p << 458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05;
      break;
    case MT::PinholeRadTanK3:
      p.resize(9);
      p << 458.654, 457.296, 367.215, 248.375, -0.28340811, 0.07395907, 0.00019359, 1.76187114e-05,
          -0.0123;
      break;
    case MT::DoubleSphere: // Basalt's TUM-VI 512 cam0
      p.resize(6);
      p << 158.28600034966976, 158.2743455478755, 254.96116578191652, 256.8894394501779,
          -0.17213086034353242, 0.5931177593944744;
      break;
    case MT::ExtendedUnified: // Basalt's EuRoC cam0
      p.resize(6);
      p << 460.76484651566466, 459.4051018049483, 365.8937161309615, 249.33499869752444,
          0.5903365915227143, 1.127468196965374;
      break;
    case MT::Unified:
      p.resize(5);
      p << 300.5, 301.2, 255.1, 256.3, 0.61;
      break;
  }
  return CameraProjection(type, p);
}

const std::vector<MT>& kAllModels = allModelTypes();

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
  EXPECT_EQ(makeModel(MT::PinholeRadTan).numParameters(), 8);
  EXPECT_EQ(makeModel(MT::PinholeRadTanK3).numDistortionParameters(), 5);
  EXPECT_EQ(makeModel(MT::DoubleSphere).numParameters(), 6);
  EXPECT_EQ(makeModel(MT::DoubleSphere).numProjectionParameters(), 4);
  EXPECT_EQ(makeModel(MT::ExtendedUnified).numDistortionParameters(), 2);
  EXPECT_EQ(makeModel(MT::Unified).numParameters(), 5);
  EXPECT_THROW(CameraProjection(MT::KannalaBrandtK3, Eigen::VectorXd::Zero(4)), std::runtime_error);
}

TEST(CameraProjection, ModelNames) {
  for (MT t : kAllModels) {
    EXPECT_EQ(modelTypeFromString(modelTypeToString(t)), t);
  }
  EXPECT_THROW(modelTypeFromString("Foo"), std::runtime_error);
  EXPECT_EQ(modelTypeFromString("Fisheye624"), MT::Fisheye624);
  EXPECT_EQ(modelTypeFromString("pinhole"), MT::Linear);
  EXPECT_EQ(modelTypeFromString("KB4"), MT::KannalaBrandtK3);
  EXPECT_EQ(modelTypeFromString("equidistant"), MT::KannalaBrandtK3);
  EXPECT_EQ(modelTypeFromString("radtan"), MT::PinholeRadTan);
  EXPECT_EQ(modelTypeFromString("ds"), MT::DoubleSphere);
  EXPECT_EQ(modelTypeFromString("eucm"), MT::ExtendedUnified);
  EXPECT_EQ(modelTypeFromString("UCM"), MT::Unified);
  for (MT t : kAllModels) {
    const auto model = makeModel(t);
    // one description entry per parameter (Fisheye62/624 have one focal length)
    const std::string d = modelTypeDescription(t);
    EXPECT_EQ(std::count(d.begin(), d.end(), ',') + 1, model.numParameters()) << d;
  }
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

// wide angle (up to ~80 degrees off axis) for the fisheye models
TEST(CameraProjection, WideAngleRoundTrip) {
  for (MT t :
       {MT::KannalaBrandtK3, MT::Fisheye624, MT::DoubleSphere, MT::ExtendedUnified, MT::Unified}) {
    const auto model = makeModel(t);
    for (const Eigen::Vector3d pt :
         {Eigen::Vector3d(0.9, 0.5, 0.2), Eigen::Vector3d(-1.0, 0.1, 0.3), Eigen::Vector3d(0.2, -2.0, 0.6)}) {
      Eigen::Matrix<double, 2, 3> J_pt;
      Eigen::Matrix<double, 2, Eigen::Dynamic> J_par(2, model.numParameters());
      const Eigen::Vector2d px = model.project(pt, J_pt, J_par);
      const Eigen::Vector3d ray = model.unproject(px);
      EXPECT_LT((ray.normalized() - pt.normalized()).norm(), 1e-8) << modelTypeToString(t);
      constexpr double kEps = 1e-7;
      for (int i = 0; i < 3; i++) {
        Eigen::Vector3d ptP = pt;
        ptP[i] += kEps;
        const Eigen::Vector2d num = (model.project(ptP) - px) / kEps;
        EXPECT_LT((num - J_pt.col(i)).norm(), 1e-3) << modelTypeToString(t) << " pt " << i;
      }
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

namespace {
Eigen::Vector2d projectWith(MT type, std::initializer_list<double> params, const Eigen::Vector3d& pt) {
  Eigen::VectorXd p(params.size());
  int i = 0;
  for (double v : params) {
    p[i++] = v;
  }
  return CameraProjection(type, p).project(pt);
}
} // namespace

// special cases in which the models coincide, a check of the formulas independent of the
// Jacobian / round trip tests
TEST(CameraProjection, ModelsCoincideInSpecialCases) {
  for (const auto& pt : testPoints()) {
    for (double z : {1.0, 0.4}) {
      const Eigen::Vector3d p(pt.x(), pt.y(), z);
      const Eigen::Vector2d lin = projectWith(MT::Linear, {300, 310, 250, 240}, p);
      // no distortion
      EXPECT_LT((projectWith(MT::PinholeRadTan, {300, 310, 250, 240, 0, 0, 0, 0}, p) - lin).norm(), 1e-9);
      EXPECT_LT((projectWith(MT::Unified, {300, 310, 250, 240, 0}, p) - lin).norm(), 1e-9);
      EXPECT_LT((projectWith(MT::ExtendedUnified, {300, 310, 250, 240, 0, 1}, p) - lin).norm(), 1e-9);
      EXPECT_LT((projectWith(MT::DoubleSphere, {300, 310, 250, 240, 0, 0}, p) - lin).norm(), 1e-9);
      // EUCM with beta = 1 is UCM, DS with xi = 0 is UCM, DS with alpha = 0 is UCM with xi
      const Eigen::Vector2d ucm = projectWith(MT::Unified, {300, 310, 250, 240, 0.6}, p);
      EXPECT_LT((projectWith(MT::ExtendedUnified, {300, 310, 250, 240, 0.6, 1}, p) - ucm).norm(), 1e-9);
      EXPECT_LT((projectWith(MT::DoubleSphere, {300, 310, 250, 240, 0, 0.6}, p) - ucm).norm(), 1e-9);
      const double xi = 0.7, alpha = xi / (1 + xi);
      EXPECT_LT(
          (projectWith(MT::DoubleSphere, {300, 310, 250, 240, xi, 0}, p) -
           projectWith(MT::Unified, {300 / (1 + xi), 310 / (1 + xi), 250, 240, alpha}, p))
              .norm(),
          1e-9);
      // radtan with k3 = 0 is radtan
      EXPECT_LT(
          (projectWith(MT::PinholeRadTanK3, {300, 310, 250, 240, -0.2, 0.05, 1e-3, -2e-3, 0}, p) -
           projectWith(MT::PinholeRadTan, {300, 310, 250, 240, -0.2, 0.05, 1e-3, -2e-3}, p))
              .norm(),
          1e-9);
    }
  }
  // values computed by OpenCV (cv2.projectPoints, EuRoC cam0 + k3 = -0.0123)
  EXPECT_LT(
      (makeModel(MT::PinholeRadTanK3).project({0.3, -0.2, 1.0}) -
       Eigen::Vector2d(499.901850263462, 160.191216201210))
          .norm(),
      1e-6);
}

// values of the independent numpy implementations in tools/viba_input/vibaio.py
TEST(CameraProjection, ReferenceValues) {
  const std::vector<std::pair<MT, std::vector<Eigen::Vector2d>>> expected = {
      {MT::DoubleSphere, {{310.041976946754, 220.171602454267}, {485.220542973967, 384.801896615704}}},
      {MT::ExtendedUnified,
       {{498.579006084704, 161.139180230073}, {927.796056981362, 560.581739433662}}},
      {MT::Unified, {{341.913004087966, 198.289846186810}, {631.890225140360, 466.115521930627}}},
      {MT::KannalaBrandtK3,
       {{309.938997491476, 220.225414669242}, {484.606079704766, 384.504503621908}}},
      {MT::Fisheye624, {{387.553936427825, 192.670870577249}, {591.635356090572, 390.464845762423}}},
  };
  const std::vector<Eigen::Vector3d> pts = {{0.3, -0.2, 1.0}, {0.9, 0.5, 0.2}};
  for (const auto& [t, px] : expected) {
    const auto model = makeModel(t);
    for (size_t i = 0; i < pts.size(); i++) {
      EXPECT_LT((model.project(pts[i]) - px[i]).norm(), 1e-8) << modelTypeToString(t) << " " << i;
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
