/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdlib>
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
    settings.preStepCallback = [&](int /* nIt */) {
      if (initSettings.recomputedPreIntegrations) {
        const int64_t nRecomp = adapter.regenerateAllPreintegrationsFromImuMeasurements();
        std::cout << fmt::format("Re-computed {} preintegrations", nRecomp) << std::endl;
      }
      adapter.updateRollingShutterData();
    };
    auto summary = opt.optimize(settings);
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
    const int rc =
        runPass(initSettings, optSettings, passInput, passOutput, rigIndexStart, rigIndexEnd);
    if (rc != 0 || last) {
      return rc;
    }
    passInput = makeNextPassInput(
        filterPathIn, passOutput, outputsDir / fmt::format("pass{}_input", pass + 1));
  }

  return 0;
}
