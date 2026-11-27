/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Eigenvalues>
#include <fmt/format.h>
#include <io/SaveDeviceTrajectory.h>
#include <io/SaveOnlineCalib.h>
#include <viba/common/Report.h>
#include <viba/common/Settings.h>
#include <viba/loop_closure/Bearing.h>
#include <viba/loop_closure/LoopClosure.h>
#include <viba/problem/PointRefinement.h>
#include <viba/problem/SingleSessionProblem.h>
#include <viba/single_session/Matcher.h>
#include <viba/single_session/SingleSessionAdapter.h>

#include <CLI/CLI.hpp>

using namespace small_thing;
using namespace visual_inertial_ba;

// One pass: load, triangulate, optimize, save.
//
// Device profile (--profile auto|aria|generic, InitSettings::applyDeviceProfile): the defaults of
// --pls-scale, --imu-half-sample and --earth-rate-up (on for Project Aria glasses, off otherwise);
// options given on the command line are kept.
//
// The optimizer's damping and stopping schedule are options (OptimizerSettings):
// --damping-good-factor, --initial-damping, --later-pass-initial-damping, --min-damping,
// --relative-cost-tolerance, --absolute-cost-tolerance, --stop-step-mm. They replace the
// environment variables VIBA_DAMPING_GOOD, VIBA_PASS_INIT_DAMPING, VIBA_PASS_ABS_TOL,
// VIBA_PASS_REL_TOL, VIBA_STOP_POS_MM and VIBA_PASS_MAX_ITERS, which have no effect any more
// (nor have VIBA_ITER_DUMP, now --dump-iterations, and the removed VIBA_STEP_DIAG,
// VIBA_STEP_EXTEND, VIBA_PCG_RESIDUAL, VIBA_YAW_KNOT_FOLD).
//
// Test hooks of the yaw-drift knots (environment, off by default):
//   VIBA_YAW_GAUGE_TEST=<rad/s> yaw-drift knots away from 0 with the variables changed to
//                               compensate, before the optimization: the histogram printed after
//                               it has to be that printed before (neutrality of the knots)
//   VIBA_YAW_KNOT_INIT=<rad/s>  starts the knots away from 0, for --verify-jacobians
//                               (SingleSessionProblem.cpp; TestLoopBundle sets it)
static int gPass = 1;
static int gOptimizerCall = 0;
static std::string gDumpDir; // --dump-iterations

// --dump-iterations <dir>: key-rig poses before every iteration and at the end of every optimizer
// call, <dir>/p<pass>_c<call>_i<iteration|end>.txt, 't[s] x y z qx qy qz qw' of the body (IMU) in
// the world
static void dumpRigPoses(const SingleSessionProblem& problem, const std::string& tag) {
  if (gDumpDir.empty()) {
    return;
  }
  const std::string& dir = gDumpDir;
  std::filesystem::create_directories(dir);
  // (the true states: the variables with the yaw-drift knots applied, if there are any)
  std::vector<std::pair<int64_t, SE3>> poses;
  for (const auto& s : problem.yawKnots_trueStates()) {
    poses.emplace_back(s.timestampUs, s.T_bodyImu_world.inverse());
  }
  std::ofstream out(
      std::filesystem::path(dir) / fmt::format("p{}_c{}_i{}.txt", gPass, gOptimizerCall, tag));
  for (const auto& [timestampUs, T_world_body] : poses) {
    const Eigen::Quaterniond q = T_world_body.unit_quaternion();
    const Vec3 p = T_world_body.translation();
    out << fmt::format(
        "{:.6f} {:.9f} {:.9f} {:.9f} {:.10f} {:.10f} {:.10f} {:.10f}\n",
        timestampUs * 1e-6, p.x(), p.y(), p.z(), q.x(), q.y(), q.z(), q.w());
  }
}

// factory IMU time offsets (instantaneous sample times) -> the pre-integration's convention
static void adjustImuTimeOffsets(SessionData& fData, const InitSettings& s) {
  if (!s.imuHalfSample && s.accelLaterMs == 0.0) {
    return;
  }
  auto& factory = fData.factoryCalibration;
  for (size_t f = 0; f < factory.calib.imuModelParameters.size(); f++) {
    // samples of this IMU (ordered as during SLAM): matched by label
    const auto& slamLabels = fData.slamInfo.imuLabels;
    size_t m = 0;
    for (size_t k = 0; k < slamLabels.size(); k++) {
      if (f < factory.imuLabels.size() && slamLabels[k] == factory.imuLabels[f]) {
        m = k;
      }
    }
    double halfPeriodSec = 0.0;
    if (s.imuHalfSample && m < fData.allImuMeasurements.size() &&
        fData.allImuMeasurements[m].size() > 100) {
      const auto& meas = fData.allImuMeasurements[m];
      std::vector<int64_t> periods;
      for (size_t i = 1; i < std::min<size_t>(meas.size(), 4000); i++) {
        periods.push_back(meas[i].timestampNs - meas[i - 1].timestampNs);
      }
      std::nth_element(periods.begin(), periods.begin() + periods.size() / 2, periods.end());
      halfPeriodSec = 0.5e-9 * periods[periods.size() / 2];
    }
    auto& p = factory.calib.imuModelParameters[f];
    const double oldGyro = p.dtReferenceGyroSec, oldAccel = p.dtReferenceAccelSec;
    p.dtReferenceGyroSec -= halfPeriodSec;
    p.dtReferenceAccelSec -= halfPeriodSec + 1e-3 * s.accelLaterMs;
    int64_t numOnline = 0;
    for (size_t o = 0; o < fData.onlineCalibration.imuLabels.size(); o++) {
      if (f >= factory.imuLabels.size() ||
          fData.onlineCalibration.imuLabels[o] != factory.imuLabels[f]) {
        continue;
      }
      for (auto& calib : fData.onlineCalibration.calibs) {
        if (o < calib.imuModelParameters.size()) {
          auto& q = calib.imuModelParameters[o];
          if (std::abs(q.dtReferenceGyroSec - oldGyro) < 1e-7 &&
              std::abs(q.dtReferenceAccelSec - oldAccel) < 1e-7) {
            q.dtReferenceGyroSec = p.dtReferenceGyroSec;
            q.dtReferenceAccelSec = p.dtReferenceAccelSec;
            numOnline++;
          }
        }
      }
    }
    std::cout << fmt::format(
                     "IMU {} time offsets: gyro {:.3f} -> {:.3f} ms, accel {:.3f} -> {:.3f} ms "
                     "(half a sample period {:.3f} ms, accelerometer later by {:.3f} ms); "
                     "{} online calibration states followed",
                     f,
                     1e3 * oldGyro,
                     1e3 * p.dtReferenceGyroSec,
                     1e3 * oldAccel,
                     1e3 * p.dtReferenceAccelSec,
                     1e3 * halfPeriodSec,
                     s.accelLaterMs,
                     numOnline)
              << std::endl;
  }
}


// ---------------------------------------------------------------------------------------------
// Loop closure: runs when loop data (--lc-matches / --lc-tracks) is given, and only then. See
// viba/loop_closure/LoopClosure.h and LC_IMPL.md in the main repository. Runs after the open-loop
// optimization of the last pass: LC-KRs and candidates, 5-point verification, pose graph stages 1
// and 2 (variant --lc-variant), the corrections put into the trajectory through the drift-knot
// transform, then the loop-closed bundle adjustment. The pose graph's correspondences are those of
// --lc-matches, or, without it, the pairs between consecutive visits of the loop tracks.
struct LcOptions {
  std::string matches; // match file (empty: derived from the loop tracks, if given)
  std::string mode = "ba"; // ba (or full): pose graph, then loop-closed BA; pg: pose graph only
  int trackPairs = 20; // without --lc-matches: correspondences per pair of visits of a loop track
  std::string variant = "poses"; // poses | knots
  double knotSec = 0.0; // knots variant: interval length; 0: the problem's drift-knot intervals
  loop::LoopClosureSettings s;
  // --lc-mode ba: the loop-closed bundle adjustment after the pose graph
  int baIterations = 50;
  int pcgIterations = 100;
  double pcgResidual = 1e-6; // relative residual of the reduced system
  int gncIterations = 5; // iterations with GNC weights on the loop pairs, then the exact form
  double gncThresholdSq = 9.0; // c^2 on the median squared whitened error of a pair
  std::string tracks; // loop tracks (ba): the loop points, instead of the pairs' matches
  bool pcgAdaptive = true; // Eisenstat-Walker tolerance of the PCG
  double pcgEtaMin = 1e-6;
  double pcgEtaMax = 0.1;
  // initial damping of the loop bundle adjustment (the pass's own one is often 1e-9; the loop
  // terms are new and the knot stage works at 1e-6); <= 0: that of the pass
  double baDamping = 1e-6;
};
static LcOptions gLc;

static void addLcOptions(CLI::App& app, LcOptions& o) {
  auto& s = o.s;
  auto& d = s.odometry;
  app.add_option(
      "--lc-matches",
      o.matches,
      "Loop closure: runs when loop data (--lc-matches / --lc-tracks) is given. File of "
      "wide-baseline correspondences for the pose graph (t_a_us,cam_a,t_b_us,cam_b,u_a,v_a,u_b,v_b; "
      "tools/viba_lc); without it they are derived from --lc-tracks. Runs after the open-loop "
      "optimization of the last pass");
  app.add_option(
      "--lc-mode",
      o.mode,
      "ba (default; also: full): pose graph stages 1 and 2, corrections into the trajectory, then "
      "the loop-closed bundle adjustment (loop points with all their observations, PCG with the "
      "factor of the banded problem as preconditioner); pg: the pose graph only (diagnostic)")
      ->check(CLI::IsMember({"ba", "full", "pg"}));
  app.add_option("--lc-ba-iters", o.baIterations, "ba: iterations");
  app.add_option("--lc-pcg-iters", o.pcgIterations, "ba: PCG iterations per step");
  app.add_option("--lc-pcg-residual", o.pcgResidual, "ba: relative residual of the PCG");
  app.add_option("--lc-gnc-iters", o.gncIterations, "ba: iterations with GNC weights on the pairs");
  app.add_option("--lc-ba-gnc", o.gncThresholdSq, "ba: GNC threshold c^2");
  app.add_option(
      "--lc-tracks",
      o.tracks,
      "Loop closure: runs when loop data (--lc-matches / --lc-tracks) is given. Loop tracks "
      "(track,visit,t_us,cam,u,v; tools/viba_lc/lc_sift_tracks.py): the loop points of the "
      "loop-closed BA with all their observations, grouped by visit (instead of the matches of the "
      "pairs); without --lc-matches also the pose graph's correspondences (see --lc-track-pairs)");
  app.add_option(
      "--lc-track-pairs",
      o.trackPairs,
      "Without --lc-matches: the pose graph's correspondences from the loop tracks, at most this "
      "many observation pairs per pair of consecutive visits of a track (default: 20)");
  app.add_option("--lc-pcg-adaptive", o.pcgAdaptive, "ba: Eisenstat-Walker PCG tolerance (else --lc-pcg-residual)");
  app.add_option("--lc-pcg-eta-min", o.pcgEtaMin, "ba: floor of the adaptive PCG tolerance");
  app.add_option("--lc-pcg-eta-max", o.pcgEtaMax, "ba: cap of the adaptive PCG tolerance");
  app.add_option("--lc-ba-damping", o.baDamping, "ba: initial damping (<= 0: that of the pass)");
  app.add_option("--lc-variant", o.variant, "poses: key-rig poses as variables; knots: drift knots only");
  app.add_option("--lc-knot-sec", o.knotSec, "knots variant: knot interval [s] (0: the drift knots' intervals)");
  app.add_option("--lc-lckr-sec", s.lckrSpacingSec, "spacing of the loop-closing key rigs [s]");
  app.add_option("--lc-radius", s.candidateRadiusM, "candidate radius [m]");
  app.add_option("--lc-min-gap", s.candidateMinGapSec, "candidate time gap [s]");
  app.add_option("--lc-partners", s.maxPartners, "verified partners per LC-KR");
  app.add_option("--lc-partner-sep", s.partnerSeparationSec, "time between partners [s]");
  app.add_option("--lc-min-matches", s.minMatches, "correspondences per camera combination");
  app.add_option("--lc-min-inliers", s.minInliers, "5-point inliers of a verified pair");
  app.add_option("--lc-ransac-thr", s.ransac.thresholdRad, "5-point inlier threshold [rad]");
  app.add_option("--lc-min-parallax-deg", s.minParallaxDeg, "below: no direction constraint");
  app.add_option("--lc-loop-yaw-sigma", s.loopYawSigma, "stage 1 loop edge, yaw [rad]");
  app.add_option("--lc-loop-dir-sigma", s.loopDirSigma, "stage 1 loop edge, direction [rad]");
  app.add_option("--lc-stage2", s.stage2, "run stage 2 (loop points)");
  app.add_option("--lc-pixel-sigma", s.pixelSigma, "stage 2 [px]");
  app.add_option("--lc-huber-px", s.huberPx, "stage 2 [px]");
  app.add_option("--lc-max-points", s.maxPointsPerPair, "stage 2 points per pair");
  app.add_option("--lc-gnc1", s.gnc1.thresholdSq, "stage 1 GNC threshold c^2");
  app.add_option("--lc-gnc2", s.gnc2.thresholdSq, "stage 2 GNC threshold c^2");
  app.add_option("--lc-knot-rate-sigma", s.knotRateSigma, "knots variant: prior [rad/s]");
  app.add_option("--lc-knot-vel-sigma", s.knotVelSigma, "knots variant: prior [m/s]");
  app.add_option("--lc-odo-yaw", d.yawPerSqrtSec, "odometry: yaw [rad/sqrt(s)]");
  app.add_option("--lc-odo-pos-h", d.posHPerSqrtSec, "odometry: horizontal [m/sqrt(s)]");
  app.add_option("--lc-odo-pos-v", d.posVPerSqrtSec, "odometry: vertical [m/sqrt(s)]");
  app.add_option("--lc-odo-pos-dist", d.posPerSqrtMeter, "odometry: [m/sqrt(m)]");
  app.add_option("--lc-odo-yaw-floor", d.yawFloor, "odometry: [rad]");
  app.add_option("--lc-odo-pos-floor", d.posFloor, "odometry: [m]");
  app.add_option("--lc-odo-ref-tracks", d.refNumTracks, "odometry: reference number of tracks");
  app.add_option("--lc-odo-tracks-exp", d.tracksExponent, "odometry: exponent of the track count");
  app.add_option("--lc-odo-ref-track-sec", d.refTrackSec, "odometry: reference track length [s]");
  app.add_option("--lc-odo-track-exp", d.trackLengthExponent, "odometry: exponent of the track length");
  app.add_option("--lc-odo-max-quality", d.maxQuality, "odometry: clamp of each quality ratio");
}

static bool runLoopBundle(
    SingleSessionProblem& problem,
    SingleSessionAdapter& adapter,
    Optimizer& opt,
    const Optimizer::Settings& baseSettings,
    const loop::LoopClosureResult& res);

static bool runLoopClosureStage(
    SingleSessionProblem& problem,
    SingleSessionAdapter& adapter,
    Optimizer& opt,
    const Optimizer::Settings& baseSettings,
    const std::filesystem::path& outputsDir) {
  using namespace visual_inertial_ba::loop;
  std::string err;
  std::unique_ptr<CsvMatchSource> src;
  if (!gLc.matches.empty()) {
    src = CsvMatchSource::load(gLc.matches, &err);
  } else {
    std::vector<LoopTrackObs> obs;
    if (loadLoopTracks(gLc.tracks, obs, &err)) {
      src = CsvMatchSource::fromLoopTracks(obs, gLc.trackPairs);
    }
  }
  if (!src) {
    std::cerr << "Loop closure: " << err << std::endl;
    return false;
  }
  const std::string srcName = !gLc.matches.empty()
      ? gLc.matches
      : fmt::format("{} (pairs of consecutive visits, at most {})", gLc.tracks, gLc.trackPairs);
  LoopClosureSettings s = gLc.s;
  if (gLc.variant == "knots") {
    s.variant = PoseGraphVariant::Knots;
  } else if (gLc.variant != "poses") {
    std::cerr << "Loop closure: unknown --lc-variant " << gLc.variant << std::endl;
    return false;
  }
  if (gLc.mode != "pg" && gLc.mode != "ba" && gLc.mode != "full") {
    std::cerr << "Loop closure: --lc-mode " << gLc.mode << " is not implemented" << std::endl;
    return false;
  }
  std::vector<KeyRig> rigs;
  for (const auto& st : problem.yawKnots_trueStates()) {
    rigs.push_back({st.rigIndex, st.timestampUs, st.T_bodyImu_world.inverse()});
  }
  if (rigs.size() < 2) {
    return false;
  }
  const int numCams = adapter.numCameras();
  const CamerasAtRig cameras = [&](int64_t rigIndex) {
    std::vector<CameraAtRig> c;
    for (int k = 0; k < numCams; k++) {
      const auto* model = problem.cameraModel_find(rigIndex, k);
      const auto* extr = problem.T_Cam_BodyImu_find(rigIndex, k);
      c.push_back({model ? &model->var.value : nullptr, extr ? extr->var.value : SE3()});
    }
    return c;
  };
  const auto spans = adapter.trackSpansUs();
  const IntervalStatsFn stats = [&spans](int64_t t0, int64_t t1) {
    IntervalStats st;
    double sum = 0.0;
    for (const auto& [a, b] : spans) {
      if (b >= t0 && a <= t1) {
        st.numTracks += 1.0;
        sum += (b - a) * 1e-6;
      }
    }
    st.meanTrackSec = st.numTracks > 0.0 ? sum / st.numTracks : 0.0;
    return st;
  };
  if (s.variant == PoseGraphVariant::Knots) {
    if (gLc.knotSec > 0.0) {
      s.knotSec = gLc.knotSec;
    } else if (problem.numYawKnots() > 0) {
      for (int64_t k = 0; k <= problem.numYawKnots(); k++) {
        s.knotTimesSec.push_back((problem.yawKnotTimestampUs(k) - rigs.front().timestampUs) * 1e-6);
      }
    }
  }
  std::cout << fmt::format(
                   "Loop closure: {} key rigs, {} correspondences in {}, variant {}",
                   rigs.size(),
                   src->numCorrespondences(),
                   srcName,
                   gLc.variant)
            << std::endl;
  const LoopClosureResult res = runLoopClosure(rigs, cameras, *src, stats, s, &std::cout);
  if (!res.ok) {
    std::cerr << "Loop closure failed: " << res.message << std::endl;
    return false;
  }
  if (!outputsDir.empty()) {
    std::filesystem::create_directories(outputsDir);
    std::ofstream pairs(outputsDir / "lc_pairs.csv");
    pairs << "rig_a,t_a_us,cam_a,rig_b,t_b_us,cam_b,inliers,parallax_deg,direction,weight1,weight2\n";
    for (size_t k = 0; k < res.pairs.size(); k++) {
      const auto& p = res.pairs[k];
      pairs << fmt::format(
          "{},{},{},{},{},{},{},{:.3f},{},{},{}\n",
          res.nodes[p.nodeA].rigIndex,
          res.nodes[p.nodeA].timestampUs,
          p.camA,
          res.nodes[p.nodeB].rigIndex,
          res.nodes[p.nodeB].timestampUs,
          p.camB,
          p.fA.size(),
          p.medianParallaxRad * 180.0 / M_PI,
          p.directionValid ? 1 : 0,
          res.weights1[k],
          k < res.weights2.size() ? res.weights2[k] : -1.0);
    }
    std::ofstream nodes(outputsDir / "lc_nodes.csv");
    nodes << "rig,t_us,yaw1_rad,x1,y1,z1,yaw2_rad,x2,y2,z2\n";
    for (size_t n = 0; n < res.nodes.size(); n++) {
      const Vec4 a = res.stage1[n], b = res.stage2.empty() ? a : res.stage2[n];
      nodes << fmt::format(
          "{},{},{:.9f},{:.6f},{:.6f},{:.6f},{:.9f},{:.6f},{:.6f},{:.6f}\n",
          res.nodes[n].rigIndex,
          res.nodes[n].timestampUs,
          a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3]);
    }
  }
  if (res.correction.knots.empty()) {
    std::cout << "Loop closure: " << res.message << "; trajectory unchanged" << std::endl;
    return true;
  }
  std::vector<SE3> T_world_body;
  std::vector<double> psi;
  correctedStates(res.correction, rigs, T_world_body, psi);
  double maxMove = 0.0, maxYaw = 0.0;
  for (size_t i = 0; i < rigs.size(); i++) {
    auto& rig = problem.inertialPose(rigs[i].rigIndex);
    maxMove = std::max(
        maxMove, (T_world_body[i].translation() - rigs[i].T_world_body.translation()).norm());
    maxYaw = std::max(maxYaw, std::abs(psi[i]));
    rig.T_bodyImu_world.value = T_world_body[i].inverse();
    rig.vel_world.value = SO3::rotZ(psi[i]) * rig.vel_world.value;
  }
  // (the drift knots are 0 here: their frozen copy follows the corrected trajectory)
  problem.yawKnots_refreeze();
  adapter.updateRollingShutterData();
  std::cout << fmt::format(
                   "Loop closure: corrections put into the trajectory, largest move {:.3f} m, "
                   "largest yaw {:.4f} deg",
                   maxMove,
                   maxYaw * 180.0 / M_PI)
            << std::endl;
  if (gLc.mode != "pg") {
    return runLoopBundle(problem, adapter, opt, baseSettings, res);
  }
  return true;
}

// Loop points from loop tracks (--lc-tracks): observations kept on key rigs (within 2 ms), visits
// with at least one of them, tracks with 2 visits at least; host = the rig of the first observation;
// triangulated from all rays on the current (corrected) trajectory, or far along the host ray if
// that fails; GNC group = the pair of 20 s bins of the first and the last visit. Returns the number of
// points (-1: file error).
static int64_t addLoopTrackPoints(
    SingleSessionProblem& problem,
    SingleSessionAdapter& adapter,
    const Mat22& sqrtH) {
  using namespace visual_inertial_ba::loop;
  std::vector<LoopTrackObs> all;
  std::string err;
  if (!loadLoopTracks(gLc.tracks, all, &err)) {
    std::cerr << "Loop tracks: " << err << std::endl;
    return -1;
  }
  std::vector<int64_t> rigTimes, rigOf;
  for (const int64_t r : problem.sortedRigIndices()) {
    rigTimes.push_back(problem.inertialPose(r).timestampUs);
    rigOf.push_back(r);
  }
  auto rigAt = [&](int64_t t) -> int64_t {
    const auto it = std::lower_bound(rigTimes.begin(), rigTimes.end(), t);
    int64_t best = -1, bestDt = 2001;
    for (auto c : {it, it == rigTimes.begin() ? rigTimes.end() : std::prev(it)}) {
      if (c != rigTimes.end() && std::llabs(*c - t) < bestDt) {
        bestDt = std::llabs(*c - t);
        best = rigOf[c - rigTimes.begin()];
      }
    }
    return best;
  };
  struct Ob {
    int64_t rig;
    int cam, visit;
    int64_t t;
    Vec2 px;
    Vec3 ray, centre; // world ray of the observation, camera centre
  };
  std::map<int64_t, std::vector<Ob>> tracks;
  int64_t numKeptObs = 0;
  for (const auto& o : all) {
    const int64_t rig = rigAt(o.tUs);
    if (rig < 0 || o.cam < 0 || o.cam >= adapter.numCameras()) {
      continue;
    }
    const auto* cam = problem.cameraModel_find(rig, o.cam);
    const auto* extr = problem.T_Cam_BodyImu_find(rig, o.cam);
    Vec3 f;
    if (cam == nullptr || extr == nullptr || !bearingFromPixel(cam->var.value, o.px, f)) {
      continue;
    }
    const SE3 T_world_cam = problem.inertialPose(rig).T_bodyImu_world.value.inverse() *
        extr->var.value.inverse();
    tracks[o.track].push_back(
        {rig, o.cam, o.visit, o.tUs, o.px, T_world_cam.so3() * f, T_world_cam.translation()});
    numKeptObs++;
  }
  std::map<std::pair<int64_t, int64_t>, int> groupOf;
  int64_t numPoints = 0, numObs = 0, numVisits = 0, numVisitsTwoRigs = 0, numFar = 0, numDropped = 0;
  for (auto& [id, obs] : tracks) {
    std::sort(obs.begin(), obs.end(), [](const Ob& a, const Ob& b) { return a.t < b.t; });
    std::map<int, std::set<int64_t>> visitRigs;
    for (const auto& o : obs) {
      visitRigs[o.visit].insert(o.rig);
    }
    if (visitRigs.size() < 2) {
      continue;
    }
    // triangulation: least squares of the distances to all rays
    Mat33 A = Mat33::Zero();
    Vec3 b = Vec3::Zero();
    for (const auto& o : obs) {
      const Mat33 P = Mat33::Identity() - o.ray * o.ray.transpose();
      A += P;
      b += P * o.centre;
    }
    const Ob& h = obs.front();
    Vec3 X = h.centre + 1000.0 * h.ray;
    const Eigen::SelfAdjointEigenSolver<Mat33> es(A);
    if (es.eigenvalues().minCoeff() > 1e-6 * es.eigenvalues().maxCoeff()) {
      const Vec3 Y = A.ldlt().solve(b);
      if ((Y - h.centre).dot(h.ray) > 0.3) {
        X = Y;
      } else {
        numFar++;
      }
    } else {
      numFar++;
    }
    // observations that do not project at the start are left out (a store with more than 3 % of
    // failing factors makes the optimizer refuse every step); the track needs 2 visits after that
    std::vector<Ob> kept;
    std::map<int, std::set<int64_t>> keptVisits;
    for (const auto& o : obs) {
      const SE3 T_cam_world = problem.T_Cam_BodyImu(o.rig, o.cam).var.value *
          problem.inertialPose(o.rig).T_bodyImu_world.value;
      const Vec3 Xc = T_cam_world * X;
      Vec2 px;
      if (Xc.z() > 0.05 * Xc.norm() &&
          problem.cameraModel(o.rig, o.cam).var.value.project(Xc.normalized(), px)) {
        kept.push_back(o);
        keptVisits[o.visit].insert(o.rig);
      }
    }
    numDropped += obs.size() - kept.size();
    if (keptVisits.size() < 2 || kept.front().rig != h.rig) {
      continue;
    }
    obs.swap(kept);
    visitRigs.swap(keptVisits);
    const int64_t lp = problem.loopPoint_addNew(X, h.rig, true);
    const int64_t bin0 = obs.front().t / 20000000, bin1 = obs.back().t / 20000000;
    const auto key = std::make_pair(bin0, bin1);
    if (groupOf.count(key) == 0) {
      const int next = int(groupOf.size());
      groupOf[key] = next;
    }
    for (const auto& o : obs) {
      problem.addLoopObservation(o.rig, o.cam, lp, o.px, sqrtH, o.visit, groupOf[key]);
      numObs++;
    }
    numPoints++;
    numVisits += visitRigs.size();
    for (const auto& [_, rigs] : visitRigs) {
      numVisitsTwoRigs += rigs.size() >= 2;
    }
  }
  std::cout << fmt::format(
                   "Loop tracks: {} observations in the file, {} on key rigs; {} loop points with "
                   "{} observations, {} visits ({} seen from 2 rigs or more), {} GNC groups, {} "
                   "points started far (no triangulation), {} observations left out (not projecting)",
                   all.size(),
                   numKeptObs,
                   numPoints,
                   numObs,
                   numVisits,
                   numVisitsTwoRigs,
                   groupOf.size(),
                   numFar,
                   numDropped)
            << std::endl;
  return numPoints;
}

// The loop-closed bundle adjustment: the inlier matches of the accepted pairs become loop points
// (one variable each, in inverse depth in the earlier rig) with an observation in each visit; their
// observations are kept out of the factorized matrix (LoopBundle.h); PCG with the factor of the
// banded problem as preconditioner; drift knots active; GNC weights on the pairs for the first
// iterations, then the exact form (weight 1 or 0).
static bool runLoopBundle(
    SingleSessionProblem& problem,
    SingleSessionAdapter& adapter,
    Optimizer& opt,
    const Optimizer::Settings& baseSettings,
    const loop::LoopClosureResult& res) {
  using namespace visual_inertial_ba::loop;
  const Mat22 sqrtH = Mat22::Identity() / gLc.s.pixelSigma;
  int64_t numPoints = 0;
  if (!gLc.tracks.empty()) {
    numPoints = addLoopTrackPoints(problem, adapter, sqrtH);
    if (numPoints < 0) {
      return false;
    }
  }
  for (size_t k = 0; k < res.pairs.size() && gLc.tracks.empty(); k++) {
    const double w = res.weights2.empty() ? res.weights1[k] : res.weights2[k];
    if (w <= 0.0) {
      continue;
    }
    const LoopPair& p = res.pairs[k];
    const int64_t a = res.nodes[p.nodeA].rigIndex, b = res.nodes[p.nodeB].rigIndex;
    const SE3 T_wa = problem.inertialPose(a).T_bodyImu_world.value.inverse() *
        problem.T_Cam_BodyImu(a, p.camA).var.value.inverse();
    const SE3 T_wb = problem.inertialPose(b).T_bodyImu_world.value.inverse() *
        problem.T_Cam_BodyImu(b, p.camB).var.value.inverse();
    const size_t n = p.fA.size();
    const size_t m = gLc.s.maxPointsPerPair > 0 ? std::min<size_t>(n, gLc.s.maxPointsPerPair) : n;
    for (size_t j = 0; j < m; j++) {
      const size_t i = j * n / m;
      // midpoint of the two rays on the corrected trajectory; far (1 km) if it fails
      const Vec3 ra = T_wa.so3() * p.fA[i], rb = T_wb.so3() * p.fB[i];
      const Vec3 d = T_wb.translation() - T_wa.translation();
      const double ab = -ra.dot(rb), det = 1.0 - ab * ab;
      double da = 1000.0;
      if (det > 1e-8) {
        const double x = (d.dot(ra) - ab * (-d.dot(rb))) / det;
        const double y = (-d.dot(rb) - ab * d.dot(ra)) / det;
        if (x > 0.3 && y > 0.3) {
          da = x;
        }
      }
      const int64_t lp = problem.loopPoint_addNew(T_wa.translation() + da * ra, a, true);
      problem.addLoopObservation(a, p.camA, lp, p.pxA[i], sqrtH, 0, int(k));
      problem.addLoopObservation(b, p.camB, lp, p.pxB[i], sqrtH, 1, int(k));
      numPoints++;
    }
  }
  if (numPoints == 0) {
    std::cout << "Loop bundle adjustment: no loop point" << std::endl;
    return true;
  }
  problem.loopVirtual_build();
  // the loop points (and virtual points) join the elimination range
  opt.unregisterAllVariables();
  problem.registerPointVariables();
  opt.registeredVariablesToEliminationRange();

  auto* store = problem.loopStore();
  Optimizer::Settings s = baseSettings;
  s.solverType = Optimizer::Solver_PCG_MatrixFree;
  s.matrixFree = store;
  s.maxNumIterations = gLc.baIterations;
  s.pcgMaxIterations = gLc.pcgIterations;
  s.pcgDesiredResidual = gLc.pcgResidual;
  if (gLc.baDamping > 0.0) {
    s.damping = gLc.baDamping;
  }
  s.pcgAdaptive = gLc.pcgAdaptive;
  s.pcgEtaMin = gLc.pcgEtaMin;
  s.pcgEtaMax = gLc.pcgEtaMax;
  // the robust kernel of the loop observations: widened while GNC runs, VI-BA's reprojection loss
  // in the exact form
  const double lossWidth = problem.reprojErrorLoss().a;
  const double lossCutoff = std::sqrt(problem.reprojErrorLoss().k2);
  const auto basePreStep = baseSettings.preStepCallback;
  double mu0 = -1.0;
  s.preStepCallback = [&, basePreStep](int it) {
    if (basePreStep) {
      basePreStep(it); // (drift knots folded, rolling-shutter data)
    }
    problem.loopVirtual_sync();
    const double c2 = gLc.gncThresholdSq;
    const auto r2 = store->pairErrors();
    int on = 0;
    if (it < gLc.gncIterations) {
      if (mu0 < 0.0) {
        double maxR2 = 0.0;
        for (const double e : r2) {
          maxR2 = std::max(maxR2, e);
        }
        mu0 = std::max(2.0 * maxR2 / c2, 1.0);
      }
      const double mu = std::max(mu0 / std::pow(1.4, it), 1.0);
      store->loss().setSize(lossWidth, lossCutoff * std::sqrt(mu));
      for (size_t k = 0; k < r2.size(); k++) {
        const double a = r2[k] < 0.0 ? 1.0 : mu * c2 / (r2[k] + mu * c2);
        store->setPairWeight(int(k), a * a);
        on += a * a > 0.5;
      }
      std::cout << fmt::format("Loop pairs: GNC mu {:.3g}, {} of {} pairs above weight 0.5", mu, on, r2.size())
                << std::endl;
    } else if (it == gLc.gncIterations) {
      store->loss().setSize(lossWidth, lossCutoff);
      for (size_t k = 0; k < r2.size(); k++) {
        const bool in = r2[k] >= 0.0 && r2[k] < c2;
        store->setPairWeight(int(k), in ? 1.0 : 0.0);
        on += in;
      }
      std::cout << fmt::format("Loop pairs: exact form, {} of {} pairs kept", on, r2.size())
                << std::endl;
    }
  };
  std::cout << fmt::format(
                   "Loop bundle adjustment: {} loop points, {} observations",
                   numPoints,
                   store->observations().size())
            << std::endl;
  gOptimizerCall++;
  opt.optimize(s);
  dumpRigPoses(problem, "end");
  adapter.applyYawKnots();
  std::cout << fmt::format("Loop bundle adjustment: loop observations cost {:.3f}", store->computeCost(false))
            << std::endl;
  return true;
}

static int runPass(
    const InitSettings& initSettings,
    const OptimizerSettings& optSettings,
    const std::filesystem::path& filterPathIn,
    const std::filesystem::path& outputsDir,
    int64_t rigIndexStart,
    int64_t rigIndexEnd,
    bool lastPass) {
  std::cout << "Loading..." << std::endl;
  SessionData fData;
  fData.load(filterPathIn, /* loadImuMeasurements = */ true);
  adjustImuTimeOffsets(fData, initSettings);

  std::cout << "Building indices..." << std::endl;
  Matcher matcher;
  matcher.buildIndices(fData);

  Optimizer opt;

  std::cout << "Creating problem..." << std::endl;
  visual_inertial_ba::SingleSessionProblem problem(opt);
  visual_inertial_ba::SingleSessionAdapter adapter(problem, fData, matcher);

  adapter.initAllVariablesAndFactors(initSettings, rigIndexStart, rigIndexEnd, nullptr);

  // needed to compute factors in pt-refinement, histograms and optionally verifying jacobians
  adapter.updateRollingShutterData();

  if (optSettings.verifyJacobians) {
    std::cout << "Verifying Jacobians..." << std::endl;
    opt.verifyJacobians();
  }

  // do point refinement
  refinePoints(opt);

  std::cout << "Showing histogram..." << std::endl;
  problem.showHistogram(false);
  if (const char* e = std::getenv("VIBA_YAW_GAUGE_TEST")) {
    // knots away from 0 and variables changed to compensate: the histogram has to be the same
    problem.yawKnots_gaugeTest(std::atof(e));
    std::cout << "Showing histogram (yaw-drift gauge test)..." << std::endl;
    problem.showHistogram(false);
  }

  if (!optSettings.dontOptimize) {
    // register points as elimination range (Schur complement trick)
    problem.registerPointVariables();
    opt.registeredVariablesToEliminationRange(); // eliminate points

    std::cout << fmt::format(
        "Optimizing:\n"
        "tot recording.: {}\n"
        "#rigs.........: {}\n"
        "#point tracks.: {}\n"
        "#cameras......: {}\n"
        "#extrinsics...: {}\n"
        "#imu calibs...: {}\n",
        microsecondsString(adapter.usedRecordingLengthUs()),
        problem.numRigVariables(),
        problem.numPointTrackVariables(),
        problem.numCameraModelVariables(),
        problem.numExtrinsicsVariables(),
        problem.numImuCalibVariables());

    auto settings = optSettings.getSettings(gPass);
    std::unordered_map<int64_t, Vec3> lastRigPositions;
    double lastMaxRigStepMm = 1e30;
    const double stopStepMm = optSettings.stopStepMm;
    if (stopStepMm > 0) {
      settings.userConvergedCallback = [&](int /* nIt */) {
        double maxStep = 0.0;
        for (const auto& s : problem.yawKnots_trueStates()) {
          const Vec3 p = s.T_bodyImu_world.inverse().translation();
          auto it = lastRigPositions.find(s.rigIndex);
          maxStep =
              std::max(maxStep, it == lastRigPositions.end() ? 1e30 : (p - it->second).norm());
          lastRigPositions[s.rigIndex] = p;
        }
        lastMaxRigStepMm = 1e3 * maxStep;
        std::cout << fmt::format("    largest key-rig position step: {:.4f} mm", lastMaxRigStepMm)
                  << std::endl;
        return lastMaxRigStepMm < stopStepMm;
      };
    }
    // the yaw-drift knots go into the trajectory before every iteration (variables = true
    // states, knots = 0, frozen copy taken again): left to accumulate over a call they take the
    // variables far from the copy and steps start to fail
    settings.preStepCallback = [&](int nIt) {
      dumpRigPoses(problem, std::to_string(nIt));
      problem.yawKnots_apply();
      if (initSettings.recomputedPreIntegrations) {
        const int64_t nRecomp = adapter.regenerateAllPreintegrationsFromImuMeasurements();
        std::cout << fmt::format("Re-computed {} preintegrations", nRecomp) << std::endl;
      }
      adapter.updateRollingShutterData();
    };
    if (problem.numScaleKnots() > 0 && initSettings.scaleKnotIterations > 0) {
      // scale-knot stage(s): a few iterations with the knots, then trajectory and points are
      // divided by them; the optimization proper runs without them
      auto stageSettings = settings;
      stageSettings.maxNumIterations = initSettings.scaleKnotIterations;
      for (int stage = 0; stage < initSettings.scaleKnotStages; stage++) {
        std::cout << fmt::format("Scale-knot stage {}", stage + 1) << std::endl;
        gOptimizerCall++;
        opt.optimize(stageSettings);
        dumpRigPoses(problem, "end");
        // (the yaw-drift knots first: the scale knots work on the variables)
        adapter.applyYawKnots();
        adapter.applyScaleKnots();
        problem.yawKnots_refreeze();
      }
      problem.scaleKnots_setActive(false);
    }
    gOptimizerCall++;
    auto summary = opt.optimize(settings);
    dumpRigPoses(problem, "end");
    adapter.applyYawKnots();
    if (lastPass && (!gLc.matches.empty() || !gLc.tracks.empty()) &&
        !runLoopClosureStage(problem, adapter, opt, settings, outputsDir)) {
      return 1;
    }
    if (optSettings.triggerDebuggingOfNonlinearities > 0) {
      std::cout << "Exiting... happy debugging!" << std::endl;
      return 0;
    }

    if (!initSettings.jsonReport.empty()) {
      writeJsonReport(initSettings.jsonReport, summary);
    }

    std::cout << "Showing histogram..." << std::endl;
    problem.showHistogram(initSettings.simpleStats);
  }

  if (initSettings.evalCalibVsFactory) {
    std::map<std::string, StatsValueContainer> allStats;
    adapter.compareCalibrationVsFactory(allStats);
    SingleSessionAdapter::showCalibrationCompResults(allStats);
  }

  if (!outputsDir.empty()) {
    std::filesystem::create_directories(outputsDir);

    saveOnlineCalib(problem, fData, matcher, outputsDir / "online_calibration.jsonl");

    saveOpenLoopTrajectory(problem, fData, matcher, outputsDir / "open_loop_framerate_trajectory.csv");

    saveCloseLoopTrajectory(problem, fData, matcher, outputsDir / "closed_loop_framerate_trajectory.csv");

    if (problem.numScaleKnots() > 0 && initSettings.scaleKnotIterations <= 0) {
      std::ofstream knots(outputsDir / "scale_knots.csv");
      knots << "timestamp_us,scale\n";
      for (int64_t i = 0; i < problem.numScaleKnots(); i++) {
        knots << fmt::format(
            "{},{:.12f}\n", problem.scaleKnotTimestampUs(i), problem.scaleKnot(i));
      }
    }
    if (problem.numGravityKnots() > 0) {
      std::ofstream knots(outputsDir / "gravity_knots.csv");
      knots << "timestamp_us,gravity_x,gravity_y,gravity_z\n";
      for (int64_t i = 0; i < problem.numGravityKnots(); i++) {
        const Vec3& g = problem.gravityKnot(i);
        knots << fmt::format(
            "{},{:.9f},{:.9f},{:.9f}\n", problem.gravityKnotTimestampUs(i), g[0], g[1], g[2]);
      }
    }
  }

  return 0;
}

// Input folder of the next pass: the given input, with the trajectory and the calibration found by
// the previous pass in place of the initial ones (everything else is linked).
static std::filesystem::path makeNextPassInput(
    const std::filesystem::path& input,
    const std::filesystem::path& prevOutput,
    const std::filesystem::path& next) {
  namespace fs = std::filesystem;
  const char* kTrajectory = "open_loop_trajectory.csv";
  const char* kCalibration = "online_calibration.jsonl";
  fs::create_directories(next);
  for (const auto& entry : fs::directory_iterator(input)) {
    const fs::path name = entry.path().filename();
    if (name == kTrajectory || name == kCalibration) {
      continue;
    }
    fs::remove(next / name);
    fs::create_symlink(fs::absolute(entry.path()), next / name);
  }
  fs::copy_file(
      prevOutput / "open_loop_framerate_trajectory.csv",
      next / kTrajectory,
      fs::copy_options::overwrite_existing);
  fs::copy_file(
      prevOutput / kCalibration, next / kCalibration, fs::copy_options::overwrite_existing);
  return next;
}

int main(int argc, const char* argv[]) {
  std::filesystem::path filterPathIn;
  std::filesystem::path outputsDir;
  InitSettings initSettings;
  OptimizerSettings optSettings;
  int64_t rigIndexStart = -1, rigIndexEnd = -1;
  int numPasses = 1;

  CLI::App app{"Test of loading filter data"};
  initSettings.addCommandLine(app);
  optSettings.addCommandLine(app);
  addLcOptions(app, gLc);
  app.add_option(
      "--dump-iterations",
      gDumpDir,
      "Diagnostic: key-rig poses before every iteration and at the end of every optimizer call, "
      "<dir>/p<pass>_c<call>_i<iteration|end>.txt ('t[s] x y z qx qy qz qw' of the body IMU in "
      "the world)");
  app.add_option("-i,--input-dir", filterPathIn, "input folder path")->required();
  app.add_option("-o,--output-dir", outputsDir, "Output folder path");
  app.add_option("--rig-start", rigIndexStart, "Index of starting rig in recording");
  app.add_option("--rig-end", rigIndexEnd, "Index of ending rig in recording");
  app.add_option(
      "--num-passes",
      numPasses,
      "Number of passes (default: 1). Each further pass starts from the trajectory and calibration "
      "of the previous one and triangulates all tracks again, so that observations rejected "
      "because they disagreed with the initial trajectory can come back. Needs --output-dir "
      "(intermediate results go to pass<k>/ and pass<k>_input/ inside it)");

  CLI11_PARSE(app, argc, argv);
  initSettings.applyDeviceProfile(app, filterPathIn);

  if (numPasses > 1 && outputsDir.empty()) {
    std::cerr << "--num-passes needs --output-dir" << std::endl;
    return 1;
  }

  std::filesystem::path passInput = filterPathIn;
  for (int pass = 1; pass <= numPasses; pass++) {
    const bool last = pass == numPasses;
    const std::filesystem::path passOutput =
        last ? outputsDir : outputsDir / fmt::format("pass{}", pass);
    if (numPasses > 1) {
      std::cout << fmt::format("=== Pass {} of {} ===", pass, numPasses) << std::endl;
    }
    gPass = pass;
    gOptimizerCall = 0;
    InitSettings passSettings = initSettings;
    if (passSettings.plsDir.empty()) {
      passSettings.plsDir = filterPathIn.string();
    }
    if (pass < initSettings.skyFromPass) {
      passSettings.skyMinDist = passSettings.skyMinDistOverBaseline = 0.0;
    }
    const int rc =
        runPass(passSettings, optSettings, passInput, passOutput, rigIndexStart, rigIndexEnd, last);
    if (rc != 0 || last) {
      return rc;
    }
    passInput = makeNextPassInput(
        filterPathIn, passOutput, outputsDir / fmt::format("pass{}_input", pass + 1));
  }

  return 0;
}
