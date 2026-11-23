/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/problem/VisualFactorRelative.h>

#include <gtest/gtest.h>

#include <cmath>
#include <random>

using namespace visual_inertial_ba;

namespace {

CameraModelParam makeCamera() {
  Eigen::VectorXd p(15); // Fisheye624: f, cx, cy, 6 radial, 2 tangential, 4 thin prism
  p << 241.1, 318.2, 238.9, -0.025, 0.098, -0.067, 0.0059, 0.0012, -0.0003, 0.0004, -0.0002, 0.0007,
      -0.0001, -0.0003, 0.0001;
  const calibration::CameraCalibration calib(
      "camera-test",
      calibration::CameraProjection::ModelType::Fisheye624,
      p,
      SE3(),
      640,
      480,
      std::nullopt,
      M_PI);
  return CameraModelParam(calib, false, false);
}

struct Scene {
  SE3 T_host_world, T_obs_world, T_cam_body;
  Vec3 ptWorld;
  Vec2 obs;
  Mat22 sqrtH;
};

// two bodies about 1 m apart looking roughly along +z of their cameras, a point `depth` meters
// ahead
Scene makeScene(std::mt19937& gen, double depth) {
  std::uniform_real_distribution<double> u(-1.0, 1.0);
  auto rnd = [&]() { return Vec3(u(gen), u(gen), u(gen)); };
  Scene s;
  const SE3 T_world_host(SO3::exp(0.3 * rnd()), 2.0 * rnd());
  const SE3 T_world_obs =
      T_world_host * SE3(SO3::exp(0.2 * rnd()), 0.5 * rnd() + Vec3(0.7, 0.0, 0.0));
  s.T_host_world = T_world_host.inverse();
  s.T_obs_world = T_world_obs.inverse();
  s.T_cam_body = SE3(SO3::exp(0.1 * rnd()), 0.05 * rnd());
  const Vec3 ptCamHost(0.3 * depth * u(gen), 0.3 * depth * u(gen), depth);
  s.ptWorld = T_world_host * (s.T_cam_body.inverse() * ptCamHost);
  s.obs = Vec2(300.0 + 20.0 * u(gen), 250.0 + 20.0 * u(gen));
  s.sqrtH << 1.3, 0.1, 0.0, 0.9;
  return s;
}

// local coordinates of the point in the host, and R0, as SingleSessionProblem::pointTrack_setHost
Vec3 toLocal(const Scene& s, bool inverseDepth, Mat33& R0) {
  const Vec3 ptHost = s.T_host_world * s.ptWorld;
  R0.setIdentity();
  if (!inverseDepth) {
    return ptHost;
  }
  const Vec3 d = ptHost.normalized();
  const Vec3 e1 = d.cross(std::abs(d[0]) < 0.9 ? Vec3::UnitX() : Vec3::UnitY()).normalized();
  R0.col(0) = e1;
  R0.col(1) = d.cross(e1);
  R0.col(2) = d;
  return Vec3(0.0, 0.0, 1.0 / ptHost.norm());
}

// what the world-coordinates visual factor computes
Vec2 worldResidual(const Scene& s, const SE3& T_obs_world, const CameraModelParam& cam) {
  Vec2 proj;
  EXPECT_TRUE(cam.project(s.T_cam_body * (T_obs_world * s.ptWorld), proj));
  return s.sqrtH * (proj - s.obs);
}

} // namespace

TEST(VisualFactorRelative, SameResidualAsWorldPoint) {
  std::mt19937 gen(37);
  const CameraModelParam cam = makeCamera();
  for (double depth : {1.5, 8.0, 300.0}) {
    for (bool inverseDepth : {false, true}) {
      const Scene s = makeScene(gen, depth);
      Mat33 R0;
      const Vec3 pt = toLocal(s, inverseDepth, R0);

      const VisualFactorHosted hosted(s.obs, s.sqrtH, inverseDepth, R0);
      const auto err = hosted(
          pt,
          s.T_host_world,
          s.T_obs_world,
          s.T_cam_body,
          cam,
          small_thing::NullRef(),
          small_thing::NullRef(),
          small_thing::NullRef(),
          small_thing::NullRef(),
          small_thing::NullRef());
      ASSERT_TRUE(err.has_value());
      EXPECT_LT((*err - worldResidual(s, s.T_obs_world, cam)).norm(), 1e-7);

      const VisualFactorHostRig hostRig(s.obs, s.sqrtH, inverseDepth, R0);
      const auto errHost = hostRig(
          pt,
          s.T_cam_body,
          cam,
          small_thing::NullRef(),
          small_thing::NullRef(),
          small_thing::NullRef());
      ASSERT_TRUE(errHost.has_value());
      EXPECT_LT((*errHost - worldResidual(s, s.T_host_world, cam)).norm(), 1e-7);
    }
  }
}

TEST(VisualFactorRelative, Jacobians) {
  std::mt19937 gen(41);
  const CameraModelParam cam = makeCamera();
  const int nCam = cam.numModelParameters();
  const double eps = 1e-6;
  for (double depth : {1.5, 8.0, 300.0}) {
    for (bool inverseDepth : {false, true}) {
      const Scene s = makeScene(gen, depth);
      Mat33 R0;
      const Vec3 pt = toLocal(s, inverseDepth, R0);
      // inverse depth is a small number: its finite difference needs a step in proportion
      const Vec3 ptStep = inverseDepth ? Vec3(eps, eps, eps * pt[2]) : Vec3(eps, eps, eps);

      const VisualFactorHosted hosted(s.obs, s.sqrtH, inverseDepth, R0);
      Mat23 Jpt;
      Mat26 Jhost, Jobs, Jext;
      Mat<2, Eigen::Dynamic> Jcam(2, nCam);
      const Vec2 e0 = *hosted(
          pt, s.T_host_world, s.T_obs_world, s.T_cam_body, cam, Jpt, Jhost, Jobs, Jext, Jcam);
      auto eval = [&](const Vec3& p, const SE3& Th, const SE3& To, const SE3& Tc) {
        return *hosted(
            p,
            Th,
            To,
            Tc,
            cam,
            small_thing::NullRef(),
            small_thing::NullRef(),
            small_thing::NullRef(),
            small_thing::NullRef(),
            small_thing::NullRef());
      };
      for (int k = 0; k < 3; k++) {
        Vec3 d = Vec3::Zero();
        d[k] = ptStep[k];
        const Vec2 num =
            (eval(pt + d, s.T_host_world, s.T_obs_world, s.T_cam_body) - e0) / ptStep[k];
        EXPECT_LT((num - Jpt.col(k)).norm(), 1e-3 * (1.0 + Jpt.col(k).norm())) << "point " << k;
      }
      for (int k = 0; k < 6; k++) { // poses are updated as exp(step) * T
        Eigen::Matrix<double, 6, 1> d = Eigen::Matrix<double, 6, 1>::Zero();
        d[k] = eps;
        const SE3 D = SE3::exp(d);
        EXPECT_LT(
            ((eval(pt, D * s.T_host_world, s.T_obs_world, s.T_cam_body) - e0) / eps - Jhost.col(k))
                .norm(),
            1e-3 * (1.0 + Jhost.col(k).norm()))
            << "host pose " << k;
        EXPECT_LT(
            ((eval(pt, s.T_host_world, D * s.T_obs_world, s.T_cam_body) - e0) / eps - Jobs.col(k))
                .norm(),
            1e-3 * (1.0 + Jobs.col(k).norm()))
            << "observer pose " << k;
        EXPECT_LT(
            ((eval(pt, s.T_host_world, s.T_obs_world, D * s.T_cam_body) - e0) / eps - Jext.col(k))
                .norm(),
            1e-3 * (1.0 + Jext.col(k).norm()))
            << "extrinsics " << k;
      }

      const VisualFactorHostRig hostRig(s.obs, s.sqrtH, inverseDepth, R0);
      Mat23 JptH;
      Mat26 JextH;
      Mat<2, Eigen::Dynamic> JcamH(2, nCam);
      const Vec2 h0 = *hostRig(pt, s.T_cam_body, cam, JptH, JextH, JcamH);
      auto evalH = [&](const Vec3& p, const SE3& Tc) {
        return *hostRig(
            p, Tc, cam, small_thing::NullRef(), small_thing::NullRef(), small_thing::NullRef());
      };
      for (int k = 0; k < 3; k++) {
        Vec3 d = Vec3::Zero();
        d[k] = ptStep[k];
        EXPECT_LT(
            ((evalH(pt + d, s.T_cam_body) - h0) / ptStep[k] - JptH.col(k)).norm(),
            1e-3 * (1.0 + JptH.col(k).norm()))
            << "point (host rig) " << k;
      }
      for (int k = 0; k < 6; k++) {
        Eigen::Matrix<double, 6, 1> d = Eigen::Matrix<double, 6, 1>::Zero();
        d[k] = eps;
        EXPECT_LT(
            ((evalH(pt, SE3::exp(d) * s.T_cam_body) - h0) / eps - JextH.col(k)).norm(),
            1e-3 * (1.0 + JextH.col(k).norm()))
            << "extrinsics (host rig) " << k;
      }
    }
  }
}

// an inverse depth of zero or slightly below is a point at or just beyond infinity: a valid
// direction, continuous with small positive values; a clearly negative one is behind the camera and
// fails
TEST(VisualFactorRelative, InverseDepthThroughInfinity) {
  std::mt19937 gen(43);
  const CameraModelParam cam = makeCamera();
  const Scene s = makeScene(gen, 50.0);
  Mat33 R0;
  toLocal(s, true, R0);
  const VisualFactorHosted hosted(s.obs, s.sqrtH, true, R0);
  auto eval = [&](double rho) {
    return hosted(
        Vec3(0.0, 0.0, rho),
        s.T_host_world,
        s.T_obs_world,
        s.T_cam_body,
        cam,
        small_thing::NullRef(),
        small_thing::NullRef(),
        small_thing::NullRef(),
        small_thing::NullRef(),
        small_thing::NullRef());
  };
  const auto atInfinity = eval(0.0);
  ASSERT_TRUE(atInfinity.has_value());
  ASSERT_TRUE(eval(1e-7).has_value());
  ASSERT_TRUE(eval(-1e-7).has_value());
  EXPECT_LT((*eval(1e-7) - *atInfinity).norm(), 1e-3);
  EXPECT_LT((*eval(-1e-7) - *atInfinity).norm(), 1e-3);
  EXPECT_FALSE(eval(-10.0).has_value());
}
