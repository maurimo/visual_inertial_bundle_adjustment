/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <CLI/CLI.hpp>
#include <fmt/format.h>
#include <nlohmann/json.hpp>
#include <viba/common/Settings.h>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>

#include <logging/Checks.h>
#define DEFAULT_LOG_CHANNEL "ViBa::Settings"
#include <logging/Log.h>

namespace visual_inertial_ba {

constexpr char CalibInitArgSpec[] =
    "imu-calib|imu-extr|imu-all|cam-intr|cam-extr|cam-all|all-extr|all";
constexpr char TrajectoryInitArgSpec[] = "pose|vel|omega|all";
constexpr char ImuEstArgSpec[] =
    "gyro-bias|accel-bias|gyro-scale|accel-scale|gyro-nonorth|accel-nonorth|\n"
    "  reference-imu-time-offset|gyro-accel-time-offset|all|all-but-time-offsets|\n"
    "  all-but-biases|all-but-biases-and-scales|all-non-orths|all-time-offsets";

void InitSettings::addCommandLine(CLI::App& app) {
  app.add_option(
         "--profile",
         profile,
         "Device profile, the defaults of the options that depend on the device (options given "
         "here are kept). aria (Project Aria glasses, LaMAria): --pls-scale 1, --imu-half-sample 1, "
         "--earth-rate-up 5.365e-5 (Zurich); generic (eg TUM-VI, EuRoC): all three 0. auto "
         "(default): aria when the factory calibration has the cameras camera-slam-left and "
         "camera-slam-right, generic otherwise")
      ->check(CLI::IsMember({"auto", "aria", "generic"}));
  app.add_option(
         "--imu-calib-rw-infl",
         imuRWinflate,
         "Inflate factor for IMU calibration random walk covariances")
      ->group("Factor Weighting");
  app.add_option(
         "--cam-intr-rw-infl",
         camIntrRWinflate,
         "Inflate factor for Cam intrinsics random walk covariances")
      ->group("Factor Weighting");
  app.add_option(
         "--imu-extr-rw-infl",
         imuExtrRWinflate,
         "Inflate factor for IMU extrinsics random walk covariances")
      ->group("Factor Weighting");
  app.add_option(
         "--cam-extr-rw-infl",
         camExtrRWinflate,
         "Inflate factor for Cam extrinsics random walk covariances")
      ->group("Factor Weighting");

  app.add_option(
         "--imu-calib-fprio-infl",
         imuFactorCalibInflate,
         "Inflate factor for IMU calibration factor calibration prior (default: 100.0)")
      ->group("Factor Weighting");
  app.add_option(
         "--cam-intr-fprio-infl",
         camIntrFactorCalibInflate,
         "Inflate factor for Cam intrinsics factor calibration prior (default: 100.0)")
      ->group("Factor Weighting");
  app.add_option(
         "--imu-extr-fprio-infl",
         imuExtrFactorCalibInflate,
         "Inflate factor for IMU extrinsics factor calibration prior (default: 100.0)")
      ->group("Factor Weighting");
  app.add_option(
         "--cam-extr-fprio-infl",
         camExtrFactorCalibInflate,
         "Inflate factor for Cam extrinsics factor calibration prior (default: 100.0)")
      ->group("Factor Weighting");
  app.add_flag_function(
         "--no-fprio",
         [&](size_t /* something */) {
           imuFactorCalibInflate = camIntrFactorCalibInflate = imuExtrFactorCalibInflate =
               camExtrFactorCalibInflate = 0.0;
         },
         "Disable all factor calibration priors")
      ->group("Factor Weighting");

  app.add_option(
         "--tracking-obs-lrad",
         trackingObsLossRadius,
         fmt::format(
             "Radius for tracking observation loss (default: {:.02f})",
             kReprojectionErrorHuberLossWidth))
      ->group("Huber Soft Loss");
  app.add_option(
         "--tracking-obs-lcut",
         trackingObsLossCutoff,
         fmt::format(
             "Cutoff for tracking observation loss (default: {:.02f})",
             kReprojectionErrorHuberLossCutoff))
      ->group("Huber Soft Loss");
  app.add_option(
         "--imu-lrad",
         imuLossRadius,
         fmt::format("Radius for IMU factor loss (default: {:.02f})", kImuErrorHuberLossWidth))
      ->group("Huber Soft Loss");
  app.add_option(
         "--imu-lcut",
         imuLossCutoff,
         fmt::format("Cutoff for IMU factor loss (default: {:.02f})", kImuErrorHuberLossCutoff))
      ->group("Huber Soft Loss");
  app.add_option(
         "--calib-constant",
         calibConstant,
         "Set to constant. One or more comma-sep of:\n  " + std::string(CalibInitArgSpec))
      ->group("Calibration");
  app.add_option(
         "--calib-factory",
         calibFactory,
         "Set calib set to (constant) factory values. One or more comma-sep of:\n  " +
             std::string(CalibInitArgSpec))
      ->group("Calibration");
  app.add_option(
         "--imu-calib-estimation-options",
         imuCalibEstimationOptions,
         "Set individual IMU calib fields to (constant) override values. Comma-sep of:\n  " +
             std::string(ImuEstArgSpec))
      ->group("Calibration");
  app.add_option(
         "--imu-calib-fields-to-factory",
         imuCalibFieldsToFactory,
         "Set individual IMU calib fields to (constant) override values. Comma-sep of:\n  " +
             std::string(ImuEstArgSpec))
      ->group("Calibration");
  app.add_flag(
         "--eval-calib-vs-factory",
         evalCalibVsFactory,
         "Evaluate deviation of calibration parameters wrt factory calibration")
      ->group("Calibration");
  app.add_flag(
         "--estimate-readout-time",
         estimateReadoutTime,
         "Estimate readout-time for rolling shutter cameras")
      ->group("Calibration");
  app.add_flag(
         "--estimate-time-offset",
         estimateTimeOffset,
         "Estimate time-offset (only for rolling shutter cameras ATM)")
      ->group("Calibration");
  app.add_flag(
         "--recompute-preint",
         recomputedPreIntegrations,
         "Recompute pre-integration, at every optimization step")
      ->group("Calibration");

  app.add_option(
         "--trajectory-constant",
         trajectoryConstant,
         "Set trajectory components to constants. Comma-sep of: " +
             std::string(TrajectoryInitArgSpec))
      ->group("Trajectory");
  app.add_option(
         "--trajectory-to-gt",
         trajectoryToGt,
         "Set trajectory components to loading from GT. Comma-sep of: " +
             std::string(TrajectoryInitArgSpec))
      ->group("Trajectory");
  app.add_option(
         "--gt-trajectory-base-name",
         gtTrajectoryBaseName,
         "'base' filename of pre-downloaded trajectory GT file (inside session data dir)")
      ->group("Trajectory");
  app.add_option(
         "--gt-trajectory-type",
         gtTrajectoryType,
         "Type of trajectory GT, as defined in MapGtReader.cpp (default: frl_synthetic_spline)\n"
         "NOTE: must support querying arbitrary timestamps, to deduce linear/angular velocities")
      ->group("Trajectory");

  app.add_flag(
         "--optimize-detector-bias",
         optimizeDetectorBias,
         "Optimize image detector bias. Note: it correlates with intrinsics principal point")
      ->group("Debugging");

  app.add_option(
         "--point-param",
         pointParam,
         "How point variables are expressed: world, or in a host rig as host (position), "
         "hostinv (inverse depth), hostmix (inverse depth when far, default)")
      ->check(CLI::IsMember({"world", "host", "hostinv", "hostmix"}))
      ->group("Point parameterization");
  app.add_option(
         "--point-far-dist",
         pointFarDist,
         "hostmix: distance from the host (meters) beyond which a point is inverse depth "
         "(default: 20)")
      ->group("Point parameterization");

  app.add_option(
         "--sky-min-dist",
         skyMinDist,
         "Leave out tracks seen above the horizon and farther than this [m] (default: 0, off)")
      ->group("Sky points");
  app.add_option(
         "--sky-min-dist-over-baseline",
         skyMinDistOverBaseline,
         "Leave out tracks seen above the horizon and farther than this many times the largest "
         "distance between two of their cameras (default: 0, off)")
      ->group("Sky points");
  app.add_option(
         "--sky-min-elevation",
         skyMinElevationDeg,
         "Elevation above the horizon [deg] from which a far track counts as sky (default: 5)")
      ->group("Sky points");
  app.add_option(
         "--imu-half-sample",
         imuHalfSample,
         "Factory IMU time offsets are instantaneous sample times: use them minus half a sample "
         "period, as the pre-integration holds a sample over the interval ending at it "
         "(default: profile, aria 1, generic 0)");
  app.add_option(
         "--accel-later-ms",
         accelLaterMs,
         "The accelerometer samples are this much later than the factory time offset says [ms]; "
         "per device (default: 0)");
  app.add_option(
         "--sky-from-pass", skyFromPass, "First pass that leaves out sky points (default: 2)")
      ->group("Sky points");

  app.add_option(
         "--gravity-knot-sec",
         gravityKnotSec,
         "Gravity as a function of time, one vector every this many seconds, interpolated "
         "(default: 0, one constant gravity)")
      ->group("Gravity");
  app.add_option(
         "--gravity-mag-std", gravityMagStd, "Prior std of a knot's magnitude [m/s2] (default: 0.05)")
      ->group("Gravity");
  app.add_option(
         "--gravity-dir-std",
         gravityDirStd,
         "Prior std of a knot's horizontal components [m/s2] (default: 0, direction held)")
      ->group("Gravity");
  app.add_option(
         "--gravity-rw-mag-std",
         gravityRwMagStd,
         "Random walk std of the magnitude between two knots [m/s2] (default: 0.005)")
      ->group("Gravity");
  app.add_option(
         "--gravity-rw-dir-std",
         gravityRwDirStd,
         "Random walk std of the horizontal components between two knots [m/s2] (default: 0)")
      ->group("Gravity");
  app.add_option(
         "--gravity-scale", gravityScale, "Nominal gravity magnitude, in units of 9.81 m/s2")
      ->group("Gravity");
  app.add_option(
         "--earth-rate-up",
         earthRateUp,
         "Rotation of the Earth about the local vertical [rad/s], 7.292e-5 * sin(latitude): depends "
         "on the location of the recording; 0: not modelled (default: profile, aria 5.365e-5, the "
         "value at 47.4 deg north (Zurich, LaMAria); generic 0)")
      ->group("Gravity");
  app.add_option(
         "--scale-knot-sec",
         scaleKnotSec,
         "Scale of the IMU's metric (gravity and integrated accelerometer together) as a function "
         "of time, one knot every this many seconds, interpolated; used in a stage at the start of "
         "each pass, after which the trajectory is divided by the knots (default: 60; 0: off)")
      ->group("Gravity");
  app.add_option("--scale-knot-std", scaleKnotStd, "Prior std of a scale knot (default: 0.05)")
      ->group("Gravity");
  app.add_option(
         "--scale-knot-rw-std",
         scaleKnotRwStd,
         "Random walk std between two scale knots (default: 0.003)")
      ->group("Gravity");
  app.add_option(
         "--scale-knot-iterations",
         scaleKnotIterations,
         "Iterations of a scale-knot stage (default: 1; 0: the knots stay in for the whole "
         "optimization and are written to scale_knots.csv)")
      ->group("Gravity");
  app.add_option(
         "--scale-knot-stages", scaleKnotStages, "Number of scale-knot stages (default: 1)")
      ->group("Gravity");
  app.add_option(
         "--yaw-knot-sec",
         yawKnotSec,
         "Yaw-drift knots: rate of turning of the trajectory about the vertical, constant on "
         "intervals of at least this many seconds (longer where the point tracks need it: no "
         "track covers more than two intervals); an over-parametrization of the trajectory, in "
         "the inertial factors and in the visual factors of hosted points for the whole "
         "optimization; open loop only (default: 30; 0: off)")
      ->group("Gravity");
  app.add_option(
         "--yaw-knot-rate-std",
         yawKnotRateStd,
         "Prior std of a yaw-drift knot [rad/s] (default: 1e-2)")
      ->group("Gravity");
  app.add_option(
         "--yaw-knot-rate-rw-std",
         yawKnotRateRwStd,
         "Random walk std between two yaw-drift knots [rad/s] (default: 0, none)")
      ->group("Gravity");
  app.add_option(
         "--yaw-knot-shift-std",
         yawKnotShiftStd,
         "> 0: the yaw-drift knots also hold a velocity [m/s] at which the positions drift, with "
         "this prior std (default: 1; 0: no shift)")
      ->group("Gravity");

  app.add_option(
         "--pls-scale",
         plsScale,
         "Parasitic light row time of the Aria SLAM cameras: scale of the model's time (1: the "
         "model, 0: off; default: profile, aria 1, generic 0; off without expo_cam<k>.csv)")
      ->group("Parasitic light");
  app.add_option("--pls-eps", plsEps, "Parasitic light sensitivity (default: 1/89)")
      ->group("Parasitic light");
  app.add_option(
         "--pls-t0-ms",
         plsT0Ms,
         "Time from the end of the exposure to the first row (default: 0.35)")
      ->group("Parasitic light");
  app.add_option("--pls-row-us", plsRowUs, "Row time of the readout (default: 19)")
      ->group("Parasitic light");
  app.add_option(
         "--pls-smear-px",
         plsSmearPx,
         "Correlation length of photometric patches [pixels] (default: 0.8; 0: no attenuation)")
      ->group("Parasitic light");
  app.add_option(
         "--pls-smear-corner-px",
         plsSmearCornerPx,
         "Correlation length of corner tracks [pixels] (default: 5; 0: no attenuation)")
      ->group("Parasitic light");
  app.add_option(
         "--pls-dir",
         plsDir,
         "Folder of expo_cam<k>.csv and obs_kinds.csv (default: the input folder)")
      ->group("Parasitic light");

  app.add_option(
      "--json-report",
      jsonReport,
      "Path for JSON output containing a report on the optimization process");
  app.add_flag("--simple-stats", simpleStats, "Simplified histogram stats");
  app.add_flag("--verbose", verbose, "verbose: print residuals");

  app.add_flag("--no-rgb", noRgb, "Map KeyRigs were not constructed using RGB images");
}

// A Project Aria (Gen1) recording: the factory calibration has the two SLAM cameras
static bool isAriaRecording(const std::filesystem::path& inputDir, std::string& why) {
  const std::filesystem::path path = inputDir / "factory_calibration.json";
  std::ifstream ifs(path);
  if (!ifs) {
    why = "no " + path.string();
    return false;
  }
  bool left = false, right = false;
  try {
    const auto json = nlohmann::json::parse(ifs);
    if (json.contains("CameraCalibrations")) {
      for (const auto& cam : json["CameraCalibrations"]) {
        const std::string label = cam.value("Label", "");
        left = left || label == "camera-slam-left";
        right = right || label == "camera-slam-right";
      }
    }
  } catch (const std::exception& e) {
    why = fmt::format("{} not readable ({})", path.string(), e.what());
    return false;
  }
  why = left && right ? "factory calibration has camera-slam-left and camera-slam-right"
                      : "factory calibration has no camera-slam-left and camera-slam-right";
  return left && right;
}

std::string InitSettings::applyDeviceProfile(
    const CLI::App& app,
    const std::filesystem::path& inputDir) {
  std::string used = profile, why = "given";
  if (profile == "auto") {
    used = isAriaRecording(inputDir, why) ? "aria" : "generic";
    why = "auto: " + why;
  }
  const bool aria = used == "aria";
  std::string set, kept;
  auto setDefault = [&](const char* name, auto& field, auto value) {
    if (app.get_option(name)->count() > 0) {
      kept += fmt::format("{}{} {}", kept.empty() ? "" : ", ", name, field);
      return;
    }
    field = value;
    set += fmt::format("{}{} {}", set.empty() ? "" : ", ", name, value);
  };
  setDefault("--pls-scale", plsScale, aria ? 1.0 : 0.0);
  setDefault("--imu-half-sample", imuHalfSample, aria);
  setDefault("--earth-rate-up", earthRateUp, aria ? 5.365e-5 : 0.0);
  std::cout << fmt::format(
                   "Device profile: {} ({}); {}{}{}",
                   used,
                   why,
                   set.empty() ? "nothing set" : set,
                   kept.empty() ? "" : "; given: ",
                   kept)
            << std::endl;
  return used;
}

using namespace small_thing;

static std::unordered_map<std::string, Optimizer::SolverType> argToSolver{
    {"direct", Optimizer::Solver_Direct},
    {"auto", Optimizer::Solver_Direct}, // (formerly: iterative from 20000 key rigs on)
    {"gauss-seidel", Optimizer::Solver_PCG_GaussSeidel},
    {"jacobi", Optimizer::Solver_PCG_Jacobi},
    {"lower-prec", Optimizer::Solver_PCG_LowerPrecSolve},
};
static const char* solverArgs = "direct|auto|gauss-seidel|jacobi|lower-prec";

void OptimizerSettings::addCommandLine(CLI::App& app, const std::string& group, bool skipAdvanced) {
  app.add_option(
         "--max-num-iterations", maxNumIterations, "Max number of iterations (default: 250)")
      ->group(group);
  app.add_flag(
         "--verify-jacobians",
         verifyJacobians,
         "Check the analytic Jacobians of all factors against finite differences before optimizing")
      ->group(group);
  app.add_option(
         "--linear-solver",
         linearSolverArg,
         "Linear solver (" + std::string(solverArgs) +
             "; default: direct, and auto is direct: no automatic switch to an iterative solver)")
      ->group(group);
  app.add_option(
         "--pcg-max-iterations",
         pcgMaxIterations,
         "Max iterations for PCG iterative solvers (Gauss-Seidel or Jacobi), default: 40")
      ->group(group);
  app.add_flag("--dont-optimize", dontOptimize, "do not optimize (to just print histograms, etc)")
      ->group(group);
  app.add_option("--num-threads", numThreads, "Number of threads to use for optimization")
      ->group(group);
  app.add_option(
         "--initial-damping", initialDamping, "Damping at the start of the first pass (default: 1e-5)")
      ->group(group);
  app.add_option(
         "--later-pass-initial-damping",
         laterPassInitialDamping,
         "Damping at the start of the passes after the first (default: 1e-9)")
      ->group(group);
  app.add_option("--min-damping", minDamping, "Lowest damping (default: 1e-11)")->group(group);
  app.add_option(
         "--damping-good-factor",
         dampingGoodFactor,
         "The damping is multiplied by this after a good step (default: 0.1)")
      ->group(group);
  app.add_option(
         "--relative-cost-tolerance",
         relativeCostTolerance,
         "Converged when the cost falls by less than this fraction (counts at the damping floor "
         "only, 3 iterations in a row; default: 1e-8)")
      ->group(group);
  app.add_option(
         "--absolute-cost-tolerance",
         absoluteCostTolerance,
         "Converged when the cost falls by less than this (as the relative tolerance; default: "
         "0.02)")
      ->group(group);
  app.add_option(
         "--stop-step-mm",
         stopStepMm,
         "Converged when an iteration moves no key rig by more than this [mm] (as the cost "
         "tolerances; default: 0.5; 0: off)")
      ->group(group);
  if (!skipAdvanced) {
    app.add_option(
           "--debug-nonlinearities-at",
           triggerDebuggingOfNonlinearities,
           "Trigger debugging of factor non linearities at iteration, default: -1 (ie disabled)")
        ->group(group);
  }
  app.callback([&]() {
    XR_CHECK(
        argToSolver.count(linearSolverArg), "Linear solver option must be one of: {}", solverArgs);
  });
}

Optimizer::Settings OptimizerSettings::getSettings(int pass) const {
  auto solverIt = argToSolver.find(linearSolverArg);
  XR_CHECK(solverIt != argToSolver.end(), "Linear solver option must be one of: {}", solverArgs);

  // (see the header for the damping and stopping schedule)
  return {
      .maxNumIterations = maxNumIterations,
      .numThreads = numThreads,
      .solverType = solverIt->second,
      .pcgMaxIterations = pcgMaxIterations,
      .absoluteCostTolerance = absoluteCostTolerance,
      .relativeCostTolerance = relativeCostTolerance,
      .damping = pass > 1 ? laterPassInitialDamping : initialDamping,
      .dampingAdjustOnGoodStep = dampingGoodFactor,
      .dampingMin = minDamping,
      .stopOnlyAtDampingFloor = true,
      .triggerDebuggingOfNonlinearities = triggerDebuggingOfNonlinearities,
  };
}

} // namespace visual_inertial_ba
