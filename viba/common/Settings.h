/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <small_thing/Optimizer.h>
#include <viba/common/Constants.h>
#include <filesystem>
#include <vector>

namespace CLI {
class App;
} // namespace CLI

namespace visual_inertial_ba {

struct InitSettings {
  void addCommandLine(CLI::App& app);

  // Device profile (--profile auto|aria|generic): defaults of the options that depend on the device.
  // aria (Project Aria Gen1 glasses, as in LaMAria): parasitic-light row time on (plsScale 1),
  // imuHalfSample on, earthRateUp 5.365e-5 (Zurich); generic: all three off (the defaults below).
  // auto: aria when the factory calibration of the input has the cameras camera-slam-left and
  // camera-slam-right, generic otherwise. Options given on the command line are kept. Call after
  // parsing; returns the profile used and prints it in one line.
  std::string applyDeviceProfile(const CLI::App& app, const std::filesystem::path& inputDir);
  std::string profile = "auto";

  std::string calibConstant;
  std::string calibFactory;
  std::string imuCalibEstimationOptions;
  std::string imuCalibFieldsToFactory;
  bool estimateReadoutTime = false;
  bool estimateTimeOffset = false;
  bool evalCalibVsFactory = false;

  std::string trajectoryConstant;
  std::string trajectoryToGt;
  std::string gtTrajectoryBaseName;
  std::string gtTrajectoryType = "frl_synthetic_spline";

  std::filesystem::path jsonReport;

  double trackingObsLossRadius = kReprojectionErrorHuberLossWidth;
  double trackingObsLossCutoff = kReprojectionErrorHuberLossCutoff;

  // how point variables are expressed: "world", or in a host rig (the rig of the middle observation
  // of the track) as "host" (position), "hostinv" (inverse depth), "hostmix" (inverse depth for the
  // points farther than pointFarDist meters from their host, position for the others)
  std::string pointParam = "hostmix";
  double pointFarDist = 20.0;

  double imuRWinflate = 1.0;
  double camIntrRWinflate = 1.0;
  double imuExtrRWinflate = 1.0;
  double camExtrRWinflate = 1.0;

  double imuFactorCalibInflate = 100.0;
  double camIntrFactorCalibInflate = 100.0;
  double imuExtrFactorCalibInflate = 100.0;
  double camExtrFactorCalibInflate = 100.0;

  bool recomputedPreIntegrations = false;

  bool optimizeDetectorBias = false;

  bool verbose = false;
  bool simpleStats = false;

  double imuLossRadius = kImuErrorHuberLossWidth;
  double imuLossCutoff = kImuErrorHuberLossCutoff;

  bool noRgb = false;

  // Sky points. A track is left out when, from all the cameras that see it, the triangulated
  // point is at least skyMinElevationDeg above the horizon (the horizon being orthogonal to
  // gravity) and far: farther than skyMinDist meters, or than skyMinDistOverBaseline times the
  // largest distance between two of its cameras (each test is off when 0). Applied from pass
  // skyFromPass on (see --num-passes), when the trajectory is good enough for "far" to mean it.
  double skyMinElevationDeg = 5.0;
  double skyMinDist = 0.0;
  double skyMinDistOverBaseline = 0.0;
  int skyFromPass = 2;

  // Aria's factory time offsets of gyroscope and accelerometer are instantaneous: the sample
  // stamped s is the value at s - offset. The pre-integration holds a sample over the interval
  // that ENDS at its (offset-corrected) stamp, so the offset to use is the factory one minus
  // half a sample period (what MPS's online calibration reports). Applied to the factory
  // calibration (prior) and to an online calibration that still equals it. On in the aria profile;
  // off in the generic one (the TUM-VI / EuRoC converters shift the IMU stamps themselves).
  bool imuHalfSample = false;
  // The accelerometer samples are this much later than the factory offset says (ms; measured
  // between the two IMUs of a device: glasses A +0.37 on imu-right, glasses B -0.18). 0: off.
  double accelLaterMs = 0.0;

  // Gravity as a function of time: one vector every gravityKnotSec seconds (0: one constant world
  // gravity), an inertial factor uses the linear interpolation at its mid time. Each knot has a
  // prior at the nominal gravity and a random walk to the next knot, with separate standard
  // deviations [m/s2] for the vertical component (magnitude) and the horizontal ones
  // (direction). A non-positive direction std holds the direction.
  double gravityKnotSec = 0.0;
  double gravityMagStd = 0.05;
  double gravityDirStd = 0.0;
  double gravityRwMagStd = 0.005;
  double gravityRwDirStd = 0.0;
  // nominal gravity magnitude = gravityScale * 9.81 (to check the scale symmetry)
  double gravityScale = 1.0;

  // Rotation of the Earth about the local vertical [rad/s] (7.292e-5 * sin(latitude); positive in
  // the northern hemisphere), taken out of the gyroscope's rotation in the inertial factors. The
  // horizontal component needs the direction of north and is left to the gyroscope bias. 0: off.
  // Location dependent: the aria profile sets the value at 47.4 deg north (Zurich, the LaMAria
  // recordings), 5.365e-5; the generic profile 0. Give the value of the place of the recording.
  double earthRateUp = 0.0;

  // Scale of the IMU's metric as a function of time: one factor every scaleKnotSec seconds (0:
  // off), linearly interpolated at the mid time of an inertial factor, that multiplies gravity and
  // the integrated accelerometer signal together (velocity and position increments). Where the
  // trajectory is too large by some factor the knots go to that factor: a slow scale drift of the
  // trajectory, which otherwise needs a coordinated move of all poses, becomes a local variable.
  // Each knot has a prior at 1 and a random walk to the next knot.
  // The knots are used in a stage at the start of each pass: scaleKnotIterations iterations with
  // the knots, then the trajectory and the points are divided by them and they are set back to 1
  // (scaleKnotStages times); the optimization proper follows without them. The knots of one
  // minute are known to about 0.1 % from the IMU: more iterations or a loose random walk put that
  // noise into the trajectory.
  // scaleKnotIterations = 0: the knots stay in for the whole optimization and are written to
  // scale_knots.csv (the caller straightens).
  double scaleKnotSec = 60.0;
  double scaleKnotStd = 0.05;
  double scaleKnotRwStd = 0.003;
  int scaleKnotIterations = 1;
  int scaleKnotStages = 1;

  // Yaw-drift knots (viba/problem/YawDrift.h): the rate [rad/s] at which the trajectory turns
  // about the vertical, constant on intervals of at least yawKnotSec seconds (0: none), longer
  // where needed for no point track to cover more than two intervals. They
  // over-parametrize the trajectory without changing the problem (the factors see the variables
  // with the drift applied), stay in for the whole optimization and are put into the trajectory
  // before every iteration. The prior (std yawKnotRateStd, at 0) only keeps them defined;
  // yawKnotRateRwStd > 0 adds a random walk between neighbours. They need the scale knots, and
  // work in open loop only (a landmark seen again much later would join far intervals), with
  // hosted points, one IMU and a free trajectory (otherwise they are left out).
  double yawKnotSec = 30.0;
  double yawKnotRateStd = 1e-2;
  double yawKnotRateRwStd = 0.0;
  // > 0: the knots also hold a velocity at which the positions drift (prior std [m/s], at 0)
  double yawKnotShiftStd = 1.0;

  // Parasitic light row time of the Aria SLAM cameras (OV7251: the storage node of a pixel keeps
  // collecting light, with relative sensitivity plsEps, until its row is read, plsT0Ms + row *
  // plsRowUs after the exposure). What row y of an image with exposure te shows is later than the
  // time stamp by D(y, te) = eps w (te + w) / 2 / (te + eps w), w = t0 + y t_row, attenuated when
  // the ghost is smeared over more than the size of a feature (plsSmearPx for photometric
  // patches, plsSmearCornerPx for corner tracks: observations of kind 0 in obs_kinds.csv).
  // The observation is taken at frame time + plsScale * D (the rolling-shutter visual factors).
  // Needs the exposure of every image: expo_cam<k>.csv (t_ns, exposure_s, ...) in plsDir (default:
  // the input folder); without these files the model is off. plsScale: 1 in the aria profile, 0
  // (off) in the generic one. The other values are those fitted on the Aria SLAM cameras.
  double plsScale = 0.0;
  double plsEps = 1.0 / 89.0;
  double plsT0Ms = 0.35;
  double plsRowUs = 19.0;
  double plsSmearPx = 0.8;
  double plsSmearCornerPx = 5.0;
  std::string plsDir;
};

struct OptimizerSettings {
  void
  addCommandLine(CLI::App& app, const std::string& group = "Optimizer", bool skipAdvanced = false);

  // pass: 1 for the first pass (--initial-damping), > 1 for the later ones
  // (--later-pass-initial-damping)
  small_thing::Optimizer::Settings getSettings(int pass = 1) const;

  int maxNumIterations = 250;
  // direct (default; "auto" is the same) or one of the iterative solvers. There is no automatic
  // switch to an iterative solver: PCG with Gauss-Seidel does not converge at low damping, and the
  // direct solver's memory grows linearly (1.7 GB at 10 130 key rigs).
  std::string linearSolverArg = "direct";
  // order in which the direct solver eliminates the reduced system (key rigs, calibration, knots,
  // after the points): "time" (default: key rigs by timestamp, a calibration window or knot after
  // the last key rig it couples, global variables last) or "amd" (BaSpaCho's fill-reducing order).
  // The problem is a band in time; on R_11_5cp with the row-time model AMD gives a factor of
  // 4.3 GB and 3 min 15 s per factorization, the time order 1.7 GB and 28 s (same iterates).
  std::string eliminationOrderArg = "time";
  int pcgMaxIterations = 40;
  bool dontOptimize = false;
  bool verifyJacobians = false;
  unsigned int numThreads = 8;

  // Damping and stopping schedule. The joint corrections of trajectory and points (scale, heading)
  // only happen once the damping is low; with the optimizer's own schedule (x0.7 per good step,
  // from 1e-5) that takes 20-30 iterations, and most of a run is spent after the result is final.
  // Hence: the damping comes down by 10 per good step to a floor of 1e-11, a later pass (which
  // starts from the result of the previous one) starts at 1e-9, and the tolerances only count once
  // the damping is at its floor (3 iterations in a row stop the optimization).
  double initialDamping = 1e-5;
  double laterPassInitialDamping = 1e-9;
  double minDamping = 1e-11;
  double dampingGoodFactor = 0.1;
  double relativeCostTolerance = 1e-8;
  double absoluteCostTolerance = 0.02;
  // an iteration that moves no key rig by more than this [mm] counts as converged (ark_vi_ba; 0:
  // off)
  double stopStepMm = 0.5;

  int triggerDebuggingOfNonlinearities = -1;
};

} // namespace visual_inertial_ba
