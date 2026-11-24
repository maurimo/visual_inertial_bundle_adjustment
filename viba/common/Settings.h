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
  double earthRateUp = 0.0;

  // Scale of the IMU's metric as a function of time: one factor every scaleKnotSec seconds (0:
  // off), linearly interpolated at the mid time of an inertial factor, that multiplies gravity and
  // the integrated accelerometer signal together (velocity and position increments). Where the
  // trajectory is too large by some factor the knots go to that factor: a slow scale drift of the
  // trajectory, which otherwise needs a coordinated move of all poses, becomes a local variable.
  // The caller then divides the trajectory by the knots and sets them back to 1. Each knot has a
  // prior at 1 and a random walk to the next knot.
  double scaleKnotSec = 0.0;
  double scaleKnotStd = 0.05;
  double scaleKnotRwStd = 0.01;
};

struct OptimizerSettings {
  void
  addCommandLine(CLI::App& app, const std::string& group = "Optimizer", bool skipAdvanced = false);

  small_thing::Optimizer::Settings getSettings(int numRigs, bool muted = false) const;

  int maxNumIterations = 250;
  std::string linearSolverArg = "auto";
  int pcgMaxIterations = 40;
  bool dontOptimize = false;
  bool verifyJacobians = false;
  unsigned int numThreads = 8;
  double initialDamping = 1e-5;
  double minDamping = 1e-9;

  int triggerDebuggingOfNonlinearities = -1;
};

} // namespace visual_inertial_ba
