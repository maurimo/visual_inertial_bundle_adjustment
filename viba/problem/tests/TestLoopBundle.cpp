/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Synthetic tests of the loop-closed bundle adjustment (LoopBundle.h):
//  - the generalized drift-knot wrap equals YawDrifted where both apply;
//  - neutrality of the knots for loop factors spanning many intervals (the mechanism of
//    VIBA_YAW_GAUGE_TEST), and that a plain (unwrapped) loop factor would not be neutral;
//  - Jacobians with the knots away from 0 (the mechanism of --verify-jacobians with
//    VIBA_YAW_KNOT_INIT);
//  - L (L^T v) on BaSpaCho's factor equals the product with the matrix that was factorized;
//  - one PCG step with the matrix-free loop term equals the dense solve of the true system;
//    with virtual points it is the same step;
//  - a short Levenberg-Marquardt run with the new solver.

#include <viba/problem/SingleSessionProblem.h>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
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

struct Options {
  double knotInit = 0.0; // VIBA_YAW_KNOT_INIT
  bool priors = false; // pose priors (well-posed linear systems; not knot-neutral)
  bool virtualPoints = false;
  bool shortLoopTwin = false; // one ordinary hosted point and one loop point, same geometry
  double perturb = 1.0; // scale of the errors of the starting values (2 cm, 3 mrad, 5 cm)
  uint32_t seed = 7;
};

struct Problem {
  std::unique_ptr<small_thing::Optimizer> opt;
  std::unique_ptr<SingleSessionProblem> prob;
  std::vector<SE3> truth; // T_bodyImu_world
  int64_t twinTrack = -1, twinLoop = -1;
};

// 121 rigs at 1 s on a ring of 8 m walked in 60 s (two visits of the same places), one camera
// looking forward; ordinary points seen over +-2 s, loop points seen around t and t + 60 s
std::unique_ptr<Problem> build(const Options& o) {
  auto P = std::make_unique<Problem>();
  P->opt = std::make_unique<small_thing::Optimizer>();
  P->prob = std::make_unique<SingleSessionProblem>(*P->opt, Muted);
  auto& prob = *P->prob;
  std::mt19937 gen(o.seed);
  std::normal_distribution<double> N(0.0, 1.0);
  std::uniform_real_distribution<double> U(0.0, 1.0);
  const int numRigs = 121;
  const int64_t t0 = 1000000;
  const SE3 T_cam_body(SO3::exp(Vec3(0.02, -0.03, 0.01)), Vec3(0.01, -0.005, 0.02));
  for (int i = 0; i < numRigs; i++) {
    const double t = double(i);
    const double phi = 2.0 * M_PI * t / 60.0;
    const double r = 8.0 + 0.7 * std::sin(2.0 * M_PI * t / 37.0);
    const Vec3 p(r * std::cos(phi), r * std::sin(phi), 0.3 * std::sin(t / 9.0));
    const Vec3 fwd(-std::sin(phi), std::cos(phi), 0.0), down(0.0, 0.0, -1.0);
    Mat33 R;
    R.col(0) = down.cross(fwd);
    R.col(1) = down;
    R.col(2) = fwd;
    const SE3 T_world_body(SO3(Eigen::Quaterniond(R).normalized()), p);
    P->truth.push_back(T_world_body.inverse());
    // the variables: the truth perturbed
    const SE3 noisy = SE3::exp(
                          o.perturb *
                          (Vec6() << 0.02 * N(gen), 0.02 * N(gen), 0.02 * N(gen), 0.003 * N(gen),
                           0.003 * N(gen), 0.003 * N(gen))
                              .finished()) *
        P->truth.back();
    prob.inertialPose_set(i, noisy, Vec3::Zero(), Vec3::Zero(), t0 + int64_t(i) * 1000000);
  }
  const int64_t camIdx = prob.cameraModel_addNew(makeCamera(), t0, -1, 0, true);
  const int64_t extIdx = prob.T_Cam_BodyImu_addNew(T_cam_body, t0, -1, 0, true);
  for (int i = 0; i < numRigs; i++) {
    prob.cameraModel_setRef(i, 0, int(camIdx));
    prob.T_Cam_BodyImu_setRef(i, 0, int(extIdx));
  }
  const CameraModelParam cam = makeCamera();
  auto project = [&](int rig, const Vec3& X, Vec2& px) {
    const Vec3 Xc = T_cam_body * (P->truth[rig] * X);
    if (Xc.z() < 0.5 || Xc.norm() > 25.0 || !cam.project(Xc, px)) {
      return false;
    }
    return px.x() > 5.0 && px.x() < 635.0 && px.y() > 5.0 && px.y() < 475.0;
  };
  auto randomLandmark = [&]() {
    const double a = 2.0 * M_PI * U(gen);
    const double rad = U(gen) < 0.3 ? 4.0 + U(gen) : 12.0 + 3.0 * U(gen);
    return Vec3(rad * std::cos(a), rad * std::sin(a), -1.5 + 4.0 * U(gen));
  };
  const Mat22 sqrtH = Mat22::Identity() * 2.0; // 0.5 px

  // ordinary points: observations within +-2 s of the host
  struct Track {
    Vec3 X;
    int host;
    std::vector<std::pair<int, Vec2>> obs;
    bool inverseDepth;
  };
  std::vector<Track> tracks;
  for (int h = 2; h + 2 < numRigs; h++) {
    for (int k = 0; k < 6; k++) {
      Track tr{randomLandmark(), h, {}, k % 2 == 0};
      for (int r = h - 2; r <= h + 2; r++) {
        Vec2 px;
        if (project(r, tr.X, px)) {
          tr.obs.emplace_back(r, px + 0.3 * Vec2(N(gen), N(gen)));
        }
      }
      bool hostSees = false;
      for (const auto& ob : tr.obs) {
        hostSees |= ob.first == h;
      }
      if (tr.obs.size() >= 3 && hostSees) {
        tracks.push_back(tr);
      }
    }
  }
  // the loop twin: an ordinary point and a loop point with the same geometry (one interval)
  Track twin{Vec3::Zero(), 30, {}, true};
  if (o.shortLoopTwin) {
    for (int attempt = 0; attempt < 1000 && twin.obs.size() < 3; attempt++) {
      twin.X = randomLandmark();
      twin.obs.clear();
      for (const int r : {30, 31, 33}) {
        Vec2 px;
        if (project(r, twin.X, px)) {
          twin.obs.emplace_back(r, px + 0.3 * Vec2(N(gen), N(gen)));
        }
      }
    }
    tracks.push_back(twin);
  }
  std::vector<std::pair<int64_t, int64_t>> spans;
  for (const auto& tr : tracks) {
    spans.emplace_back(
        prob.inertialPose(tr.obs.front().first).timestampUs,
        prob.inertialPose(tr.obs.back().first).timestampUs);
  }
  if (o.knotInit != 0.0) {
    setenv("VIBA_YAW_KNOT_INIT", std::to_string(o.knotInit).c_str(), 1);
  }
  prob.yawKnots_init(10.0, spans, 1.0 / 0.01, 0.0, 1.0 / 1.0);
  unsetenv("VIBA_YAW_KNOT_INIT");
  // all point tracks first (the factors keep pointers to them)
  for (const auto& tr : tracks) {
    const int64_t idx = prob.pointTrack_addNew(tr.X + o.perturb * 0.05 * Vec3(N(gen), N(gen), N(gen)));
    prob.pointTrack_setHost(idx, tr.host, tr.inverseDepth);
  }
  for (size_t k = 0; k < tracks.size(); k++) {
    for (const auto& [r, px] : tracks[k].obs) {
      prob.addVisualFactor(r, 0, int64_t(k), px, sqrtH, prob.reprojErrorLoss());
    }
  }
  if (o.shortLoopTwin) {
    P->twinTrack = int64_t(tracks.size()) - 1;
    P->twinLoop = prob.loopPoint_addNew(prob.pointTrack_world(P->twinTrack), twin.host, true);
    for (const auto& [r, px] : twin.obs) {
      prob.addLoopObservation(r, 0, P->twinLoop, px, sqrtH, 0, 0);
    }
  }
  // loop points: seen from 3 rigs around t and from 3 rigs around t + 60 s
  int pair = 0;
  for (int t = 8; t + 61 < numRigs; t += 6, pair++) {
    for (int k = 0; k < 4; k++) {
      const Vec3 X = randomLandmark();
      std::vector<std::pair<int, Vec2>> v1, v2;
      for (int r = t - 1; r <= t + 1; r++) {
        Vec2 px;
        if (project(r, X, px)) {
          v1.emplace_back(r, px + 0.3 * Vec2(N(gen), N(gen)));
        }
        if (project(r + 60, X, px)) {
          v2.emplace_back(r + 60, px + 0.3 * Vec2(N(gen), N(gen)));
        }
      }
      if (v1.size() < 2 || v2.size() < 2) {
        continue;
      }
      const int64_t lp =
          prob.loopPoint_addNew(X + o.perturb * 0.05 * Vec3(N(gen), N(gen), N(gen)), v1.front().first, k % 2 == 0);
      for (const auto& [r, px] : v1) {
        prob.addLoopObservation(r, 0, lp, px, sqrtH, 0, pair);
      }
      for (const auto& [r, px] : v2) {
        prob.addLoopObservation(r, 0, lp, px, sqrtH, 1, pair);
      }
    }
  }
  if (o.virtualPoints) {
    prob.loopVirtual_build();
  }
  if (o.priors) {
    Mat66 H = Mat66::Identity();
    H.topLeftCorner<3, 3>() *= 1.0 / (0.1 * 0.1);
    H.bottomRightCorner<3, 3>() *= 1.0 / (0.02 * 0.02);
    for (int i = 0; i < numRigs; i++) {
      prob.addPosePrior(i, P->truth[i], H);
    }
  }
  prob.registerPointVariables();
  P->opt->registeredVariablesToEliminationRange();
  return P;
}

// cost of the visual factors and of the loop observations (the knot priors left out: they are
// not invariant, by design)
double visualCost(const small_thing::Optimizer& opt) {
  double c = 0.0;
  for (const auto& [_, st] : opt.factorStores.stores) {
    const std::string name = st->name();
    if (name.find("VisualFactor") != std::string::npos ||
        name.find("LoopObservationStore") != std::string::npos) {
      if (st->role() != small_thing::FactorStoreBase::kRolePreconditionerOnly) {
        c += st->computeCost(false);
      }
    }
  }
  return c;
}

int maxSpanIntervals(SingleSessionProblem& prob) {
  int m = 0;
  for (const auto& o : prob.loopStore()->observations()) {
    m = std::max(m, o.kHi - o.kLo + 1);
  }
  return m;
}

} // namespace

TEST(LoopBundle, WrapEqualsYawDrifted) {
  Options o;
  o.knotInit = 2e-3;
  o.shortLoopTwin = true;
  auto P = build(o);
  auto& prob = *P->prob;
  // the twin's observations from rigs other than the host, both ways
  std::vector<Vec2> plain, loop;
  for (const auto& [_, st] : P->opt->factorStores.stores) {
    if (st->name().find("YawDrifted<visual_inertial_ba::VisualFactorHosted") == std::string::npos ||
        st->role() != small_thing::FactorStoreBase::kRoleNormal) {
      continue;
    }
    for (int64_t k = 0; k < st->numCosts(); k++) {
      if (st->costVar(k, 0) == &prob.pointTrack(P->twinTrack)) {
        Eigen::VectorXd e(2);
        st->unweightedError(k, e);
        plain.push_back(e);
      }
    }
  }
  const auto* store = prob.loopStore();
  for (int64_t k = 0; k < store->numCosts(); k++) {
    const auto& ob = store->observations()[k];
    if (ob.point == P->twinLoop && ob.kHi >= ob.kLo) {
      Eigen::VectorXd e(2);
      store->unweightedError(k, e);
      loop.push_back(e);
    }
  }
  ASSERT_EQ(plain.size(), loop.size());
  ASSERT_GE(plain.size(), 1u);
  for (size_t k = 0; k < plain.size(); k++) {
    EXPECT_LT((plain[k] - loop[k]).norm(), 1e-9) << plain[k].transpose() << " / " << loop[k].transpose();
  }
}

TEST(LoopBundle, GaugeNeutralManyIntervals) {
  Options o;
  auto P = build(o);
  auto& prob = *P->prob;
  const int span = maxSpanIntervals(prob);
  std::cout << "knots " << prob.numYawKnots() << ", loop observations span up to " << span
            << " knot intervals" << std::endl;
  ASSERT_GE(span, 5);
  const double before = visualCost(*P->opt);
  prob.yawKnots_gaugeTest(3.4 / 60.0 * M_PI / 180.0); // 3.4 deg/min (+ 30 x that in m/s)
  const double after = visualCost(*P->opt);
  std::cout << "visual + loop cost " << before << " -> " << after << " with the knots moved"
            << std::endl;
  EXPECT_NEAR(after, before, 1e-9 * before);
  // without the wrap the same move is not neutral (the test sees it)
  prob.yawKnots_setActive(false);
  const double plain = visualCost(*P->opt);
  std::cout << "  the same without the knots in the factors: " << plain << std::endl;
  EXPECT_GT(std::abs(plain - before), 1e-3 * before);
  prob.yawKnots_setActive(true);
}

TEST(LoopBundle, Jacobians) {
  Options o;
  o.knotInit = 1e-3; // as --verify-jacobians with VIBA_YAW_KNOT_INIT
  o.virtualPoints = true;
  auto P = build(o);
  ASSERT_GE(maxSpanIntervals(*P->prob), 5);
  std::string failures;
  const bool ok = P->opt->verifyJacobians(1e-7, 1e-3, 1e-3, false, [&](const std::string& m) {
    if (m.find("FAILED") != std::string::npos || m.find("LoopObservationStore") != std::string::npos) {
      std::cout << m;
    }
    if (m.find("FAILED") != std::string::npos) {
      failures += m;
    }
  });
  EXPECT_TRUE(ok) << failures;
}

TEST(LoopBundle, FactorProduct) {
  Options o;
  o.priors = true;
  auto P = build(o);
  auto& opt = *P->opt;
  auto solver = opt.initSolver(1, true);
  const auto& acc = solver->accessor();
  Eigen::VectorXd grad = Eigen::VectorXd::Zero(solver->order());
  Eigen::VectorXd hess = Eigen::VectorXd::Zero(solver->dataSize());
  opt.computeGradHess(grad.data(), acc, hess.data(), false, false, nullptr);
  opt.addDamping(hess, acc, 1e-4);
  const int64_t start = opt.elimRanges.back();
  solver->factorUpTo(hess.data(), start);
  std::mt19937 gen(3);
  std::normal_distribution<double> N(0.0, 1.0);
  Eigen::VectorXd v(solver->order());
  for (int i = 0; i < v.size(); i++) {
    v[i] = N(gen);
  }
  const int64_t off = solver->spanVectorOffset(start);
  Eigen::VectorXd ref = Eigen::VectorXd::Zero(solver->order());
  solver->addMvFrom(hess.data(), start, v.data(), solver->order(), ref.data(), solver->order(), 1);
  solver->factorFrom(hess.data(), start);
  Eigen::VectorXd out = Eigen::VectorXd::Zero(solver->order());
  small_thing::addFactorProductFrom(*solver, hess.data(), start, v.data(), out.data());
  const double rel = (out - ref).tail(solver->order() - off).norm() / ref.tail(solver->order() - off).norm();
  std::cout << "L (L^T v) against M v: relative difference " << rel << " (reduced size "
            << solver->order() - off << ")" << std::endl;
  EXPECT_LT(rel, 1e-10);
  // the same with 4 threads
  dispenso::ThreadPool pool(4);
  Eigen::VectorXd out4 = Eigen::VectorXd::Zero(solver->order());
  small_thing::addFactorProductFrom(*solver, hess.data(), start, v.data(), out4.data(), &pool);
  const double rel4 = (out4 - out).tail(solver->order() - off).norm() / out.tail(solver->order() - off).norm();
  std::cout << "4 threads against 1: relative difference " << rel4 << std::endl;
  EXPECT_LT(rel4, 1e-13);
}

namespace {
small_thing::Optimizer::Settings pcgSettings(SingleSessionProblem& prob) {
  small_thing::Optimizer::Settings s;
  s.numThreads = 1;
  s.solverType = small_thing::Optimizer::Solver_PCG_MatrixFree;
  s.matrixFree = prob.loopStore();
  s.pcgMaxIterations = 500;
  s.pcgDesiredResidual = 1e-13;
  s.log = nullptr;
  return s;
}
} // namespace

TEST(LoopBundle, PcgStepEqualsDenseSolve) {
  for (const double damping : {1e-4, 1e-8}) {
    Options o;
    o.priors = true;
    auto P = build(o);
    auto& prob = *P->prob;
    auto r = P->opt->linearSolveForTest(pcgSettings(prob), damping, /* densify = */ true);
    prob.loopStore()->addTrueHessianDense(r.dense, damping);
    const Eigen::VectorXd ref = r.dense.ldlt().solve(r.grad);
    const double rel = (r.step - ref).norm() / ref.norm();
    std::cout << "damping " << damping << ": PCG step against the dense solve of the true system: "
              << rel << "\n  " << r.report << std::endl;
    EXPECT_LT(rel, 1e-7);
  }
}

TEST(LoopBundle, VirtualPointsSameStep) {
  // the same true system with and without virtual points: only the preconditioner changes
  Options o;
  o.priors = true;
  auto A = build(o);
  o.virtualPoints = true;
  auto B = build(o);
  // (small: the damping of the rigs is relative to the diagonal of the factorized matrix, which
  // the virtual points change)
  const double damping = 1e-10;
  auto ra = A->opt->linearSolveForTest(pcgSettings(*A->prob), damping);
  auto rb = B->opt->linearSolveForTest(pcgSettings(*B->prob), damping);
  std::cout << "without virtual points:\n  " << ra.report << "\nwith virtual points:\n  "
            << rb.report << std::endl;
  // compare the steps of the rigs (the vector orders of the two problems differ)
  auto rigStep = [](small_thing::Optimizer& opt, SingleSessionProblem& prob,
                    const Eigen::VectorXd& step) {
    auto solver = opt.initSolver(1, true);
    Eigen::VectorXd out(6 * prob.numRigVariables());
    for (int64_t i = 0; i < prob.numRigVariables(); i++) {
      out.segment<6>(6 * i) =
          step.segment<6>(solver->accessor().paramStart(prob.inertialPose(i).T_bodyImu_world.index));
    }
    return out;
  };
  const Eigen::VectorXd sa = rigStep(*A->opt, *A->prob, ra.step);
  const Eigen::VectorXd sb = rigStep(*B->opt, *B->prob, rb.step);
  const double rel = (sa - sb).norm() / sa.norm();
  std::cout << "rig steps: relative difference " << rel << std::endl;
  EXPECT_LT(rel, 1e-5);
}

TEST(LoopBundle, LevenbergMarquardt) {
  // (errors of the start small enough for the truncated Huber loss of the observations: beyond
  // 3 sigma an observation has no gradient and does not come back)
  Options o;
  o.priors = true;
  o.virtualPoints = true;
  o.perturb = 0.25;
  auto P = build(o);
  auto& prob = *P->prob;
  auto s = pcgSettings(prob);
  s.maxNumIterations = 15;
  s.pcgDesiredResidual = 1e-8;
  s.pcgMaxIterations = 200;
  s.preStepCallback = [&](int) {
    prob.yawKnots_apply();
    prob.loopVirtual_sync();
  };
  const auto summary = P->opt->optimize(s);
  std::cout << "LM with the matrix-free loop term: cost " << summary.initialCost << " -> "
            << summary.finalCost << " in " << summary.numIterations << " iterations" << std::endl;
  // noise floor: 0.3 px of noise against a sigma of 0.5 px, 2 components per observation
  int64_t numObs = 0;
  for (const auto& [_, st] : P->opt->factorStores.stores) {
    const std::string name = st->name();
    if ((name.find("VisualFactor") != std::string::npos ||
         name.find("LoopObservationStore") != std::string::npos) &&
        st->role() != small_thing::FactorStoreBase::kRolePreconditionerOnly) {
      numObs += st->numCosts();
    }
  }
  const double floor = 0.5 * 2.0 * (0.3 / 0.5) * (0.3 / 0.5) * numObs;
  std::cout << "noise floor of the cost about " << floor << " (" << numObs << " observations)"
            << std::endl;
  EXPECT_LT(summary.finalCost, 1.3 * floor);
  // the rigs near the truth
  double worst = 0.0;
  for (int64_t i = 0; i < prob.numRigVariables(); i++) {
    worst = std::max(
        worst,
        (prob.inertialPose(i).T_bodyImu_world.value.inverse().translation() -
         P->truth[i].inverse().translation())
            .norm());
  }
  // (printed only: without inertial factors the scale and the drift of this problem are held by
  // the pose priors alone, 10 cm, so a few cm from the truth are within the model)
  std::cout << "largest rig position error " << worst << " m" << std::endl;
}

// the adaptive PCG tolerance (Eisenstat-Walker) against the fixed one: the same end, fewer PCG
// iterations in all; 4 threads for the product
TEST(LoopBundle, LevenbergMarquardtAdaptivePcg) {
  double cost[2];
  int pcgIters[2];
  for (const bool adaptive : {false, true}) {
    Options o;
    o.priors = true;
    o.virtualPoints = true;
    o.perturb = 0.25;
    auto P = build(o);
    auto& prob = *P->prob;
    auto s = pcgSettings(prob);
    s.numThreads = 4;
    s.maxNumIterations = 15;
    s.pcgDesiredResidual = 1e-8;
    s.pcgMaxIterations = 200;
    s.pcgAdaptive = adaptive;
    s.pcgEtaMin = 1e-8;
    s.pcgEtaMax = 0.1;
    int total = 0;
    s.log = [&](const std::string& m) {
      const size_t at = m.find("iters: ");
      if (at != std::string::npos) {
        total += std::atoi(m.c_str() + at + 7);
      }
    };
    s.preStepCallback = [&](int) {
      prob.yawKnots_apply();
      prob.loopVirtual_sync();
    };
    const auto summary = P->opt->optimize(s);
    cost[adaptive] = summary.finalCost;
    pcgIters[adaptive] = total;
    std::cout << (adaptive ? "adaptive" : "fixed 1e-8") << ": cost " << summary.initialCost << " -> "
              << summary.finalCost << ", " << summary.numIterations << " iterations, " << total
              << " PCG iterations in all" << std::endl;
  }
  EXPECT_NEAR(cost[1], cost[0], 1e-3 * cost[0]);
  EXPECT_LT(pcgIters[1], pcgIters[0]);
}
