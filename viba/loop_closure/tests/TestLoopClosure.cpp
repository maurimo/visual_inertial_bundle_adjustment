/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Synthetic tests of the loop-closure stages 1-3: the 5-point solver (scale-free translation),
// the drift mapping, the pose graph in both variants (recovery of an injected yaw and position
// drift, rejection of a wrong pair), and the Jacobians of stages 1 and 2 (verifyJacobians).

#include <viba/loop_closure/Bearing.h>
#include <viba/loop_closure/LoopClosure.h>

#include <gtest/gtest.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>

using namespace visual_inertial_ba;
using namespace visual_inertial_ba::loop;

namespace {

std::shared_ptr<CameraModelParam> makeCamera() {
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
  return std::make_shared<CameraModelParam>(calib, false, false);
}

Mat33 rotY(double a) {
  Mat33 R;
  R << std::cos(a), 0.0, std::sin(a), 0.0, 1.0, 0.0, -std::sin(a), 0.0, std::cos(a);
  return R;
}

double angleOf(const Mat33& R) {
  return std::acos(std::clamp(0.5 * (R.trace() - 1.0), -1.0, 1.0));
}

// ---------------------------------------------------------------------------------------------
// 5-point

struct TwoView {
  std::vector<Vec3> f1, f2;
  Mat33 R;
  Vec3 t; // with its true length
};

TwoView makeTwoView(std::mt19937& gen, int n, double baseline, double noiseRad, double outliers) {
  std::normal_distribution<double> N(0.0, 1.0);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  TwoView v;
  v.R = SO3::exp(0.4 * Vec3(N(gen), N(gen), N(gen))).matrix();
  v.t = baseline * Vec3(N(gen), N(gen), 0.3 * N(gen)).normalized();
  for (int k = 0; k < n; k++) {
    const Vec3 X1(4.0 * N(gen), 3.0 * N(gen), 6.0 + 8.0 * U(gen));
    const Vec3 X2 = v.R * X1 + v.t;
    if (X2.z() < 0.5) {
      continue;
    }
    Vec3 a = X1.normalized(), b = X2.normalized();
    a = (a + noiseRad * Vec3(N(gen), N(gen), N(gen))).normalized();
    b = (b + noiseRad * Vec3(N(gen), N(gen), N(gen))).normalized();
    if (U(gen) < outliers) {
      b = Vec3(N(gen), N(gen), 3.0 + N(gen)).normalized();
    }
    v.f1.push_back(a);
    v.f2.push_back(b);
  }
  return v;
}

} // namespace

TEST(FivePoint, MinimalExact) {
  std::mt19937 gen(5);
  for (int trial = 0; trial < 50; trial++) {
    const TwoView v = makeTwoView(gen, 5, 1.0, 0.0, 0.0);
    if (v.f1.size() < 5) {
      continue;
    }
    const auto Es = fivePointEssential(v.f1.data(), v.f2.data());
    Mat33 Etrue = SO3::hat(v.t.normalized()) * v.R;
    Etrue /= Etrue.norm();
    double best = 1e9;
    for (const auto& E : Es) {
      best = std::min({best, (E - Etrue).norm(), (E + Etrue).norm()});
    }
    EXPECT_LT(best, 1e-6) << "trial " << trial << ", " << Es.size() << " solutions";
  }
}

// the translation comes out as a unit direction whatever its true length: the same relative pose
// scaled by 0.2 or by 20 gives the same direction
TEST(FivePoint, RansacScaleFreeTranslation) {
  for (const double scale : {0.2, 1.0, 20.0}) {
    std::mt19937 gen(11);
    TwoView v = makeTwoView(gen, 150, 1.0, 0.0005, 0.25);
    // scale the scene and the translation together: the bearings do not change
    const Vec3 tTrue = scale * v.t;
    RansacSettings rs;
    rs.thresholdRad = 0.003;
    const RansacResult r = ransacRelativePose(v.f1, v.f2, rs);
    ASSERT_TRUE(r.ok) << "scale " << scale;
    EXPECT_GT(r.inliers.size(), 90u);
    EXPECT_LT(angleOf(r.pose.R * v.R.transpose()), 2e-3);
    EXPECT_NEAR(r.pose.t.norm(), 1.0, 1e-9);
    EXPECT_GT(r.pose.t.dot(tTrue.normalized()), std::cos(0.02)) << "scale " << scale;
    // the direction of camera 2 seen from camera 1
    const Vec3 c2 = -(v.R.transpose() * tTrue);
    EXPECT_GT(r.pose.directionInCamera1().dot(c2.normalized()), std::cos(0.02));
  }
}

// the bearings of the 5-point stage: the round trip of unprojectNoChecks with project is off near
// the border of the image (printed), that of bearingFromPixel is not
TEST(FivePoint, CameraRoundTrip) {
  const auto cam = makeCamera();
  double worstPlain = 0.0, worst = 0.0;
  Vec2 where = Vec2::Zero();
  int failed = 0;
  double minFailRadius = 1e9;
  // image radius of rays at 88 deg from the axis (the smallest over the azimuths): beyond it,
  // project() may refuse (z <= 0 at 90 deg)
  double maxRadius = 1e9;
  for (double az = 0.0; az < 2.0 * M_PI; az += 0.05) {
    const double th = 88.0 * M_PI / 180.0;
    Vec2 px;
    if (cam->project(Vec3(std::sin(th) * std::cos(az), std::sin(th) * std::sin(az), std::cos(th)), px)) {
      maxRadius = std::min(maxRadius, (px - Vec2(318.2, 238.9)).norm());
    }
  }
  for (double u = 5.0; u < 636.0; u += 5.0) {
    for (double v = 5.0; v < 476.0; v += 5.0) {
      Vec2 px;
      if (cam->project(cam->unprojectNoChecks(Vec2(u, v)).normalized(), px) &&
          (px - Vec2(u, v)).norm() > worstPlain) {
        worstPlain = (px - Vec2(u, v)).norm();
        where = Vec2(u, v);
      }
      // (project() refuses rays at 90 deg and more from the axis: the corners of this camera;
      // counted as failures only within about 86 deg: 1.5 rad times the focal length)
      Vec3 f;
      if (!bearingFromPixel(*cam, Vec2(u, v), f) || !cam->project(f, px)) {
        const double r = (Vec2(u, v) - Vec2(318.2, 238.9)).norm();
        minFailRadius = std::min(minFailRadius, r);
        failed += r < maxRadius;
        continue;
      }
      worst = std::max(worst, (px - Vec2(u, v)).norm());
    }
  }
  std::cout << "round trip: unprojectNoChecks up to " << worstPlain << " px (at "
            << where.transpose() << "), bearingFromPixel up to " << worst << " px, " << failed
            << " pixels without a bearing within " << maxRadius
            << " px of the centre (88 deg), the nearest failure at " << minFailRadius << " px"
            << std::endl;
  EXPECT_LT(worst, 0.01);
  EXPECT_EQ(failed, 0);
}

// ---------------------------------------------------------------------------------------------
// drift mapping

TEST(DriftMapping, ReproducesNodes) {
  std::mt19937 gen(3);
  std::normal_distribution<double> N(0.0, 1.0);
  std::vector<int64_t> t;
  std::vector<Vec3> p;
  Vec3 x = Vec3::Zero();
  for (int i = 0; i < 600; i++) {
    t.push_back(1000000 + int64_t(i) * 100000);
    x += Vec3(0.1 + 0.02 * N(gen), 0.03 * N(gen), 0.005 * N(gen));
    p.push_back(x);
  }
  const std::vector<int> nodeRig{0, 120, 250, 400, 599};
  std::vector<double> yaw{0.0, 0.01, -0.02, 0.015, 0.03};
  std::vector<Vec3> pos;
  for (size_t k = 0; k < nodeRig.size(); k++) {
    pos.push_back(p[nodeRig[k]] + (k == 0 ? Vec3::Zero() : Vec3(0.3 * N(gen), 0.3 * N(gen), 0.1 * N(gen))));
  }
  const DriftCorrection c = fitDriftCorrection(t, p, nodeRig, yaw, pos);
  std::vector<double> psi;
  std::vector<Vec3> q;
  driftStates(c, psi, q);
  for (size_t k = 0; k < nodeRig.size(); k++) {
    EXPECT_NEAR(psi[nodeRig[k]], yaw[k], 1e-12);
    EXPECT_LT((q[nodeRig[k]] - pos[k]).norm(), 1e-9);
  }
}

// ---------------------------------------------------------------------------------------------
// synthetic recording with laps around a ring of landmarks

namespace {

struct Scene {
  std::shared_ptr<CameraModelParam> cam;
  std::vector<SE3> T_cam_body; // two cameras
  std::vector<KeyRig> truth, drifted;
  std::vector<Vec3> landmarks;
  DriftCorrection injected;
  CsvMatchSource matches;
};

std::unique_ptr<Scene> makeScene() {
  auto sc = std::make_unique<Scene>();
  std::mt19937 gen(17);
  std::normal_distribution<double> N(0.0, 1.0);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  sc->cam = makeCamera();
  for (const double side : {1.0, -1.0}) {
    const Mat33 R_body_cam = rotY(side * 0.5);
    const Vec3 l(side * 0.05, 0.0, 0.0);
    sc->T_cam_body.emplace_back(SO3(R_body_cam.transpose()), -(R_body_cam.transpose() * l));
  }
  // 200 s at 10 Hz, laps of 60 s on a ring of radius 10 m whose radius and height change slowly
  const double lapSec = 60.0;
  for (int i = 0; i <= 2000; i++) {
    const double t = 0.1 * i;
    const double phi = 2.0 * M_PI * t / lapSec;
    const double r = 10.0 + 1.0 * std::sin(2.0 * M_PI * t / 77.0);
    const Vec3 p(r * std::cos(phi), r * std::sin(phi), 0.2 * std::sin(2.0 * M_PI * t / 45.0));
    const Vec3 fwd(-std::sin(phi), std::cos(phi), 0.0); // body z: forward
    const Vec3 down(0.0, 0.0, -1.0); // body y: down
    Mat33 R;
    R.col(0) = down.cross(fwd);
    R.col(1) = down;
    R.col(2) = fwd;
    const Mat33 tilt = SO3::exp(0.02 * Vec3(std::sin(t), std::cos(0.7 * t), 0.0)).matrix();
    KeyRig k;
    k.rigIndex = i;
    k.timestampUs = 5000000 + int64_t(i) * 100000;
    k.T_world_body = SE3(SO3(Eigen::Quaterniond(tilt * R).normalized()), p);
    sc->truth.push_back(k);
  }
  // the open-loop trajectory: the truth with a yaw drift and a position drift on [60, 200] s
  {
    DriftCorrection& c = sc->injected;
    c.path.active = c.path.shift = true;
    for (const auto& k : sc->truth) {
      c.path.timeSec.push_back((k.timestampUs - sc->truth.front().timestampUs) * 1e-6);
      c.path.refPos.push_back(k.T_world_body.translation());
    }
    c.path.knotTimeSec = {0.0, 60.0, 200.0};
    const double rate = (1.5 * M_PI / 180.0) / 140.0; // 1.5 deg over 140 s
    c.knots = {Vec4::Zero(), Vec4(rate, 0.002, -0.001, 0.0005)};
    std::vector<SE3> T;
    std::vector<double> psi;
    correctedStates(c, sc->truth, T, psi);
    sc->drifted = sc->truth;
    for (size_t i = 0; i < T.size(); i++) {
      sc->drifted[i].T_world_body = T[i];
    }
  }
  // landmarks on an inner and an outer wall
  for (int k = 0; k < 3000; k++) {
    const double a = 2.0 * M_PI * U(gen);
    const double rad = k % 3 == 0 ? 5.0 + 0.5 * U(gen) : 15.0 + 2.0 * U(gen);
    sc->landmarks.emplace_back(rad * std::cos(a), rad * std::sin(a), -1.5 + 4.0 * U(gen));
  }
  // matches between images every 2 s, more than 30 s apart and within 4 m (truth), 10 % wrong
  auto project = [&](const KeyRig& k, int c, const Vec3& X, Vec2& px) {
    const Vec3 Xc = sc->T_cam_body[c] * (k.T_world_body.inverse() * X);
    if (Xc.z() < 0.5 || Xc.norm() > 30.0) {
      return false;
    }
    if (!sc->cam->project(Xc, px)) {
      return false;
    }
    return px.x() > 10.0 && px.x() < 630.0 && px.y() > 10.0 && px.y() < 470.0;
  };
  for (size_t b = 0; b < sc->truth.size(); b += 20) {
    for (size_t a = 0; a + 300 < b; a += 20) {
      const auto& A = sc->truth[a];
      const auto& B = sc->truth[b];
      if ((A.T_world_body.translation() - B.T_world_body.translation()).norm() > 4.0) {
        continue;
      }
      for (int ca = 0; ca < 2; ca++) {
        for (int cb = 0; cb < 2; cb++) {
          for (const Vec3& X : sc->landmarks) {
            Vec2 pa, pb;
            if (!project(A, ca, X, pa) || !project(B, cb, X, pb)) {
              continue;
            }
            pa += 0.3 * Vec2(N(gen), N(gen));
            pb += 0.3 * Vec2(N(gen), N(gen));
            if (U(gen) < 0.1) {
              pb = Vec2(20.0 + 600.0 * U(gen), 20.0 + 440.0 * U(gen));
            }
            sc->matches.add(A.timestampUs, ca, B.timestampUs, cb, pa, pb);
          }
        }
      }
    }
  }
  return sc;
}

const Scene& scene() {
  static const std::unique_ptr<Scene> sc = makeScene();
  return *sc;
}

CamerasAtRig camerasOf(const Scene& sc) {
  return [&sc](int64_t) {
    std::vector<CameraAtRig> c;
    for (const auto& T : sc.T_cam_body) {
      c.push_back({sc.cam.get(), T});
    }
    return c;
  };
}

struct Errors {
  double endYawDeg, maxPosM;
};
Errors errorsOf(const std::vector<KeyRig>& truth, const std::vector<SE3>& est) {
  Errors e{0.0, 0.0};
  for (size_t i = 0; i < truth.size(); i++) {
    e.maxPosM = std::max(e.maxPosM, (truth[i].T_world_body.translation() - est[i].translation()).norm());
  }
  e.endYawDeg = 180.0 / M_PI *
      std::abs(yawOf(truth.back().T_world_body.so3().matrix() * est.back().so3().inverse().matrix()));
  return e;
}

// true rotation camera A -> camera B of a pair
Mat33 trueRelative(const Scene& sc, const LoopClosureResult& res, const LoopPair& p) {
  const SE3& A = sc.truth[res.nodeRig[p.nodeA]].T_world_body;
  const SE3& B = sc.truth[res.nodeRig[p.nodeB]].T_world_body;
  return (sc.T_cam_body[p.camB].so3() * B.so3().inverse() * A.so3() *
          sc.T_cam_body[p.camA].so3().inverse())
      .matrix();
}

RelativePose truePose(const Scene& sc, const LoopClosureResult& res, const LoopPair& p) {
  const SE3& A = sc.truth[res.nodeRig[p.nodeA]].T_world_body;
  const SE3& B = sc.truth[res.nodeRig[p.nodeB]].T_world_body;
  const SE3 T_ba = sc.T_cam_body[p.camB] * B.inverse() * A * sc.T_cam_body[p.camA].inverse();
  RelativePose r;
  r.R = T_ba.so3().matrix();
  r.t = T_ba.translation().normalized();
  return r;
}

double rmsEpipolar(const RelativePose& pose, const LoopPair& p) {
  const Mat33 E = SO3::hat(pose.t) * pose.R;
  double s2 = 0.0;
  for (size_t k = 0; k < p.fA.size(); k++) {
    const double e = epipolarAngularError(E, p.fA[k], p.fB[k]);
    s2 += e * e;
  }
  return std::sqrt(s2 / std::max<size_t>(p.fA.size(), 1));
}

void runDriftRecovery(PoseGraphVariant variant, bool stage2) {
  const Scene& sc = scene();
  std::vector<SE3> before;
  for (const auto& k : sc.drifted) {
    before.push_back(k.T_world_body);
  }
  const Errors e0 = errorsOf(sc.truth, before);
  LoopClosureSettings s;
  s.variant = variant;
  s.stage2 = stage2;
  s.knotSec = 30.0;
  // the loop edges as accurate as the synthetic matches make them, and an odometry model that
  // allows the injected drift (0.18 deg and 3 cm per 20 s)
  s.loopYawSigma = 0.001;
  s.loopDirSigma = 0.01;
  s.odometry.yawPerSqrtSec = 1e-3;
  s.odometry.posHPerSqrtSec = s.odometry.posVPerSqrtSec = 1e-2;
  const LoopClosureResult res =
      runLoopClosure(sc.drifted, camerasOf(sc), sc.matches, nullptr, s, &std::cout);
  ASSERT_TRUE(res.ok) << res.message;
  ASSERT_GE(res.numVerified, 5);
  std::vector<SE3> after;
  std::vector<double> psi;
  correctedStates(res.correction, sc.drifted, after, psi);
  const Errors e1 = errorsOf(sc.truth, after);
  std::cout << "variant " << (variant == PoseGraphVariant::Knots ? "knots" : "rig poses")
            << (stage2 ? ", stage 2" : ", stage 1") << ": " << res.nodes.size() << " nodes, "
            << res.numVerified << " pairs, " << res.numPoints << " points; end yaw error "
            << e0.endYawDeg << " -> " << e1.endYawDeg << " deg, largest position error "
            << e0.maxPosM << " -> " << e1.maxPosM << " m" << std::endl;
  const auto& st = stage2 ? res.stage2 : res.stage1;
  for (size_t n = 0; n < res.nodes.size(); n++) {
    const auto& tr = sc.truth[res.nodeRig[n]].T_world_body;
    const auto& dr = sc.drifted[res.nodeRig[n]].T_world_body;
    std::cout << "  node " << n << " t " << (res.nodes[n].timestampUs - sc.truth[0].timestampUs) * 1e-6
              << " s: yaw error " << 180.0 / M_PI * (yawOf(tr.so3().matrix() * dr.so3().inverse().matrix()) - st[n][0])
              << " deg, position error " << (tr.translation() - st[n].tail<3>()).norm() << " m" << std::endl;
  }
  EXPECT_GT(e0.endYawDeg, 1.4);
  EXPECT_LT(e1.endYawDeg, 0.15);
  EXPECT_LT(e1.maxPosM, 0.5 * e0.maxPosM);
  // No wrong pair in this scene, but one low-parallax pair (0.48 deg) comes out of the 5-point
  // stage 0.5 deg off in rotation (see LC_IMPL.md, open questions) and GNC switches it off: at
  // most one rejected pair is accepted here; the rejected ones are printed.
  int rejected = 0;
  for (size_t k = 0; k < res.pairs.size(); k++) {
    const auto& p = res.pairs[k];
    if (res.weights1[k] == 1.0) {
      continue;
    }
    rejected++;
    std::cout << "rejected pair " << k << ": t " << res.nodes[p.nodeA].timestampUs
                                    << " / " << res.nodes[p.nodeB].timestampUs << ", "
                                    << p.fA.size() << " inliers, parallax "
                                    << p.medianParallaxRad * 180.0 / M_PI << " deg, fit sigmas "
                                    << p.sigmaRotationRad << " / " << p.sigmaDirectionRad
                                    << ", rotation error against the truth "
                                    << 180.0 / M_PI * angleOf(p.pose.R * trueRelative(sc, res, p).transpose())
                                    << " deg; rms epipolar error of the estimate "
                                    << rmsEpipolar(p.pose, p) << ", of the truth "
                                    << rmsEpipolar(truePose(sc, res, p), p) << std::endl;
  }
  EXPECT_LE(rejected, 1);
}

} // namespace

TEST(PoseGraph, Stage1DriftRecoveryRigPoses) {
  runDriftRecovery(PoseGraphVariant::RigPoses, false);
}

TEST(PoseGraph, Stage1DriftRecoveryKnots) {
  runDriftRecovery(PoseGraphVariant::Knots, false);
}

TEST(PoseGraph, Stage2DriftRecoveryRigPoses) {
  runDriftRecovery(PoseGraphVariant::RigPoses, true);
}

TEST(PoseGraph, Stage2DriftRecoveryKnots) {
  runDriftRecovery(PoseGraphVariant::Knots, true);
}

// a wrong loop pair (its relative rotation turned by 6 deg about the vertical) is switched off by
// GNC, in both variants
TEST(PoseGraph, GncRejectsWrongPair) {
  const Scene& sc = scene();
  LoopClosureSettings s;
  s.stage2 = false;
  s.loopYawSigma = 0.001;
  s.loopDirSigma = 0.01;
  const LoopClosureResult res =
      runLoopClosure(sc.drifted, camerasOf(sc), sc.matches, nullptr, s, nullptr);
  ASSERT_TRUE(res.ok);
  ASSERT_GE(res.pairs.size(), 5u);
  for (const auto variant : {PoseGraphVariant::RigPoses, PoseGraphVariant::Knots}) {
    std::unique_ptr<PoseGraph> g;
    if (variant == PoseGraphVariant::Knots) {
      KnotLayout L;
      for (const auto& r : sc.drifted) {
        L.rigTimesUs.push_back(r.timestampUs);
        L.rigPos.push_back(r.T_world_body.translation());
      }
      L.knotTimeSec = uniformKnotTimes(200.0, 30.0);
      L.nodeRig = res.nodeRig;
      g = std::make_unique<PoseGraph>(res.nodes, L);
    } else {
      g = std::make_unique<PoseGraph>(res.nodes);
    }
    OdometryNoiseModel odo;
    odo.yawPerSqrtSec = 1e-3;
    odo.posHPerSqrtSec = odo.posVPerSqrtSec = 1e-2;
    for (int n = 0; n + 1 < int(res.nodes.size()); n++) {
      const double dt = (res.nodes[n + 1].timestampUs - res.nodes[n].timestampUs) * 1e-6;
      const auto sig = odo.sigmas(dt, dt, IntervalStats{odo.refNumTracks, odo.refTrackSec});
      g->addOdometry(n, n + 1, sig.yaw, sig.pos);
    }
    auto addEdge = [&](const LoopPair& p) {
      const double kr = s.fitInflation * p.sigmaRotationRad, kd = s.fitInflation * p.sigmaDirectionRad;
      return g->addLoopEdge(
          p,
          std::sqrt(s.loopYawSigma * s.loopYawSigma + kr * kr),
          std::sqrt(s.loopDirSigma * s.loopDirSigma + kd * kd));
    };
    for (const auto& p : res.pairs) {
      addEdge(p);
    }
    LoopPair wrong = res.pairs[res.pairs.size() / 2];
    // (about the camera's y axis, which is the vertical for these cameras)
    wrong.pose.R = rotY(6.0 * M_PI / 180.0) * wrong.pose.R;
    const int gw = addEdge(wrong);
    ASSERT_TRUE(g->solve(s.gnc1));
    EXPECT_EQ(g->groupWeight(gw), 0.0) << "error of the wrong pair " << g->groupError(gw);
    int accepted = 0;
    for (int k = 0; k < gw; k++) {
      accepted += g->groupWeight(k) > 0.0;
    }
    EXPECT_GE(accepted, gw - 1);
  }
}

// Jacobians of all factors of stages 1 and 2, nodes and knots away from 0
TEST(PoseGraph, Jacobians) {
  const Scene& sc = scene();
  LoopClosureSettings s;
  s.stage2 = false;
  const LoopClosureResult res =
      runLoopClosure(sc.drifted, camerasOf(sc), sc.matches, nullptr, s, nullptr);
  ASSERT_TRUE(res.ok);
  ASSERT_GE(res.pairs.size(), 3u);
  std::mt19937 gen(23);
  std::normal_distribution<double> N(0.0, 1.0);
  for (const auto variant : {PoseGraphVariant::RigPoses, PoseGraphVariant::Knots}) {
    for (const bool stage2 : {false, true}) {
      std::unique_ptr<PoseGraph> g;
      if (variant == PoseGraphVariant::Knots) {
        KnotLayout L;
        for (const auto& r : sc.drifted) {
          L.rigTimesUs.push_back(r.timestampUs);
          L.rigPos.push_back(r.T_world_body.translation());
        }
        L.knotTimeSec = uniformKnotTimes(200.0, 30.0);
        L.nodeRig = res.nodeRig;
        g = std::make_unique<PoseGraph>(res.nodes, L);
        VecX k(g->knots().size());
        for (int n = 0; n < k.size(); n++) {
          k[n] = n % 4 == 0 ? 3e-4 * N(gen) : 3e-3 * N(gen);
        }
        g->setStartKnots(k);
      } else {
        g = std::make_unique<PoseGraph>(res.nodes);
        std::vector<double> yaw;
        std::vector<Vec3> pos;
        for (int n = 0; n < g->numNodes(); n++) {
          yaw.push_back(0.02 * N(gen));
          pos.push_back(g->position(n) + 0.2 * Vec3(N(gen), N(gen), N(gen)));
        }
        g->setStart(yaw, pos);
      }
      for (int n = 0; n + 1 < g->numNodes(); n++) {
        g->addOdometry(n, n + 1, 0.003, Vec3(0.05, 0.05, 0.03));
      }
      for (const auto& p : res.pairs) {
        if (stage2) {
          LoopPair q = p;
          q.fA.resize(std::min<size_t>(q.fA.size(), 10));
          q.fB.resize(q.fA.size());
          q.pxA.resize(q.fA.size());
          q.pxB.resize(q.fA.size());
          g->addLoopPoints(q, 1.0, 2.0);
        } else {
          g->addLoopEdge(p, 0.005, 0.02);
        }
      }
      std::string failures;
      const bool ok = g->optimizer().verifyJacobians(
          1e-7, 2e-3, 1e-3, false, [&](const std::string& m) {
            if (m.find("FAILED") != std::string::npos) {
              failures += m;
            }
          });
      EXPECT_TRUE(ok) << (variant == PoseGraphVariant::Knots ? "knots" : "rig poses")
                      << (stage2 ? " stage 2\n" : " stage 1\n") << failures;
    }
  }
}
