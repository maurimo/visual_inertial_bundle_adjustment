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

#include <fmt/format.h>
#include <io/SaveDeviceTrajectory.h>
#include <io/SaveOnlineCalib.h>
#include <viba/common/Report.h>
#include <viba/common/Settings.h>
#include <viba/problem/PointRefinement.h>
#include <viba/problem/SingleSessionProblem.h>
#include <viba/single_session/Matcher.h>
#include <viba/single_session/SingleSessionAdapter.h>

#include <CLI/CLI.hpp>

using namespace small_thing;
using namespace visual_inertial_ba;

// One pass: load, triangulate, optimize, save.
// Experiments on the speed of convergence (environment, all off by default):
//   VIBA_ITER_DUMP=<dir>        key-rig poses before every iteration and at the end of every
//                               optimizer call: <dir>/p<pass>_c<call>_i<iteration|end>.txt,
//                               't[s] x y z qx qy qz qw' of the body (IMU) in the world
//   VIBA_PASS_MAX_ITERS=a,b,..  iteration limit of pass 1, 2, .. (missing: --max-num-iterations)
//   VIBA_PASS_REL_TOL=a,b,..    relative cost tolerance of pass 1, 2, .. (missing: the default)
//   VIBA_PASS_ABS_TOL=a,b,..    absolute cost tolerance of pass 1, 2, .. (missing: the default)
//   VIBA_PASS_INIT_DAMPING=a,.. damping at the start of pass 1, 2, .. (missing: --initial-damping)
//   VIBA_YAW_KNOT_FOLD=0        yaw-drift knots put into the trajectory at the end of an optimizer
//                               call only (default: before every iteration)
//   VIBA_YAW_GAUGE_TEST=<rad/s> yaw-drift knots away from 0 with the variables changed to
//                               compensate, before the optimization: the histogram printed after
//                               it has to be that printed before
//   (VIBA_YAW_KNOT_INIT=<rad/s> starts the knots away from 0, for --verify-jacobians)
//   VIBA_STOP_POS_MM=a,b,..     pass 1, 2, ..: an iteration that moves no key rig by more than this
//                               (position, mm) counts as converged (3 in a row at the damping floor
//                               stop the pass, as for the cost tolerances); 0 or missing: off
static int gPass = 1;
static int gOptimizerCall = 0;

static void dumpRigPoses(const SingleSessionProblem& problem, const std::string& tag) {
  const char* dir = std::getenv("VIBA_ITER_DUMP");
  if (dir == nullptr) {
    return;
  }
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

static double nthOf(const char* list, int index, double fallback) {
  std::stringstream ss(list);
  std::string item;
  for (int i = 0; std::getline(ss, item, ','); i++) {
    if (i == index && !item.empty()) {
      return std::atof(item.c_str());
    }
  }
  return fallback;
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

static int runPass(
    const InitSettings& initSettings,
    const OptimizerSettings& optSettings,
    const std::filesystem::path& filterPathIn,
    const std::filesystem::path& outputsDir,
    int64_t rigIndexStart,
    int64_t rigIndexEnd) {
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

    auto settings = optSettings.getSettings(problem.numRigVariables());
    if (const char* e = std::getenv("VIBA_PASS_MAX_ITERS")) {
      settings.maxNumIterations = (int)nthOf(e, gPass - 1, settings.maxNumIterations);
    }
    if (const char* e = std::getenv("VIBA_PASS_REL_TOL")) {
      settings.relativeCostTolerance = nthOf(e, gPass - 1, settings.relativeCostTolerance);
    }
    if (const char* e = std::getenv("VIBA_PASS_ABS_TOL")) {
      settings.absoluteCostTolerance = nthOf(e, gPass - 1, settings.absoluteCostTolerance);
    }
    if (const char* e = std::getenv("VIBA_PASS_INIT_DAMPING")) {
      settings.damping = nthOf(e, gPass - 1, settings.damping);
    }
    std::unordered_map<int64_t, Vec3> lastRigPositions;
    double lastMaxRigStepMm = 1e30;
    const char* stopPosEnv = std::getenv("VIBA_STOP_POS_MM");
    const double stopPosMm = stopPosEnv ? nthOf(stopPosEnv, gPass - 1, 0.0) : 0.0;
    if (stopPosMm > 0) {
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
        return lastMaxRigStepMm < stopPosMm;
      };
    }
    // the yaw-drift knots go into the trajectory before every iteration (variables = true
    // states, knots = 0, frozen copy taken again): left to accumulate over a call they take the
    // variables far from the copy and steps start to fail. VIBA_YAW_KNOT_FOLD=0: only at the end
    // of a call.
    const bool foldYawKnots =
        !(std::getenv("VIBA_YAW_KNOT_FOLD") && std::atoi(std::getenv("VIBA_YAW_KNOT_FOLD")) == 0);
    settings.preStepCallback = [&](int nIt) {
      dumpRigPoses(problem, std::to_string(nIt));
      if (foldYawKnots) {
        problem.yawKnots_apply();
      }
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
        runPass(passSettings, optSettings, passInput, passOutput, rigIndexStart, rigIndexEnd);
    if (rc != 0 || last) {
      return rc;
    }
    passInput = makeNextPassInput(
        filterPathIn, passOutput, outputsDir / fmt::format("pass{}_input", pass + 1));
  }

  return 0;
}
