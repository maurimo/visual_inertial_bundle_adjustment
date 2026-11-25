/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
#include <viba/common/Enumerate.h>
#include <viba/common/Settings.h>
#include <viba/problem/Histograms.h>
#include <viba/single_session/SingleSessionAdapter.h>

#include <algorithm>
#include <limits>

#include <logging/Checks.h>
#define DEFAULT_LOG_CHANNEL "ViBa::SingleSessionAdapter"
#include <logging/Log.h>

namespace visual_inertial_ba {

SingleSessionAdapter::SingleSessionAdapter(
    SingleSessionProblem& prob,
    const SessionData& fData,
    const Matcher& matcher,
    Verbosity verbosity)
    : verbosity_(verbosity),
      prob_(prob),
      fData_(fData),
      matcher_(matcher),
      numRigsInRecording_(matcher_.rigIndexToEvolvingStateIndex.size()),
      numCameras_(fData_.slamInfo.cameraSerialNumbers.size()),
      numImus_(fData_.slamInfo.imuLabels.size()) {
  if (verbosity_ != Muted) {
    XR_LOGI("#IMUs: {}, #cameras: {}", numImus_, numCameras_);
  }
}

void SingleSessionAdapter::initNonRigParams(const ParamInitSettings& settings) {
  initCamIntrinsics(
      settings.camIntrConstant,
      settings.estimateReadoutTime,
      settings.estimateTimeOffset,
      settings.camIntrToFactory);

  initCamExtrinsics(settings.camExtrConstant, settings.camExtrToFactory);

  initImuCalibs(
      settings.imuCalibConstant,
      settings.imuCalibToFactory,
      settings.imuCalibEstimationOptions,
      settings.imuCalibSetToFactory);

  if (numImus_ > 1) {
    initImuExtrinsics(settings.imuExtrConstant, settings.imuExtrToFactory);
  }

  // note: we need to update rolling shutter data
  // - after intrinsics are initialized
  // - before points are triangulated
  updateRollingShutterData();

  if (verbosity_ != Muted) {
    XR_LOGI("Initializing tracking points from observation (re-triangulating)...");
  }
  initPointsFromObservations();
}

void SingleSessionAdapter::initAllVariablesAndFactors(
    const InitSettings& s,
    int64_t rigIndexStart,
    int64_t rigIndexEnd,
    const TrajectoryBase* trajectory) {
  ParamInitSettings paramInitSettings{
      .rigIndexStart = rigIndexStart,
      .rigIndexEnd = rigIndexEnd,
  };
  paramInitSettings.setTrajectoryConstants(s.trajectoryConstant);
  paramInitSettings.setTrajectoryToGt(s.trajectoryToGt);
  paramInitSettings.setCalibConstants(s.calibConstant);
  paramInitSettings.setCalibToFactory(s.calibFactory);
  paramInitSettings.setImuCalibEstimationOptions(s.imuCalibEstimationOptions);
  paramInitSettings.setImuCalibToFactory(s.imuCalibFieldsToFactory);
  paramInitSettings.estimateReadoutTime = s.estimateReadoutTime;
  paramInitSettings.estimateTimeOffset = s.estimateTimeOffset;
  if (verbosity_ != Muted) {
    paramInitSettings.printImuCalibOptions();
  }
  skyMinElevationDeg_ = s.skyMinElevationDeg;
  skyMinDist_ = s.skyMinDist;
  skyMinDistOverBaseline_ = s.skyMinDistOverBaseline;
  plsInit(s);

  if (!paramInitSettings.poseToGt && !paramInitSettings.velToGt && !paramInitSettings.omegaToGt) {
    initAllParams(paramInitSettings);
  } else {
    initAllParamsWithGtTrajectory(paramInitSettings, trajectory);
  }

  // yaw-drift knots: only where every factor with a pose sees them (see YawDrift.h)
  const char* noYawKnots = nullptr;
  if (s.pointParam == "world") {
    noYawKnots = "points in world coordinates";
  } else if (numImus_ > 1) {
    noYawKnots = "more than one IMU";
  } else if (s.scaleKnotSec <= 0.0) {
    noYawKnots = "no scale knots (their inertial factor is the one that is wrapped)";
  } else if (paramInitSettings.poseConstant || paramInitSettings.velConstant) {
    noYawKnots = "a constant trajectory";
  }
  if (s.yawKnotSec > 0.0 && noYawKnots != nullptr && verbosity_ != Muted) {
    XR_LOGI("Yaw-drift knots: off ({})", noYawKnots);
  }
  if (s.yawKnotSec > 0.0 && noYawKnots == nullptr) {
    // (before the factors: the visual factors of hosted points and the inertial factors use them)
    std::vector<std::pair<int64_t, int64_t>> trackSpansUs;
    for (const auto& observations : pointTrackObservations_) {
      int64_t firstUs = std::numeric_limits<int64_t>::max(), lastUs = 0;
      for (const auto& obsIndex : observations) {
        const int64_t t = fData_.trackingObservations[obsIndex].captureTimestampUs;
        firstUs = std::min(firstUs, t);
        lastUs = std::max(lastUs, t);
      }
      if (!observations.empty()) {
        trackSpansUs.emplace_back(firstUs, lastUs);
      }
    }
    prob_.yawKnots_init(
        s.yawKnotSec,
        trackSpansUs,
        1.0 / s.yawKnotRateStd,
        s.yawKnotRateRwStd > 0.0 ? 1.0 / s.yawKnotRateRwStd : 0.0,
        s.yawKnotShiftStd > 0.0 ? 1.0 / s.yawKnotShiftStd : 0.0);
  }

  addVisualFactors(
      s.trackingObsLossRadius,
      s.trackingObsLossCutoff,
      s.optimizeDetectorBias,
      s.pointParam,
      s.pointFarDist);

  if (prob_.numYawKnots() > 0 && verbosity_ != Muted) {
    auto lengths = prob_.yawKnotIntervalsSec();
    std::sort(lengths.begin(), lengths.end());
    const auto longer = [&](double sec) {
      return lengths.end() - std::upper_bound(lengths.begin(), lengths.end(), sec);
    };
    XR_LOGI(
        "Yaw-drift knots: {} knots, rate constant on intervals of {:.1f} to {:.1f} s (median "
        "{:.1f}; at least {} s and long enough for the tracks that start in the interval before); "
        "intervals longer than 40 / 60 / 120 s: {} / {} / {}; prior std {} rad/s, "
        "random walk std {} rad/s, shift velocity std {} m/s (0: no shift); {} observations more "
        "than one interval from their host rig (plain factors)",
        prob_.numYawKnots(),
        lengths.front(),
        lengths.back(),
        lengths[lengths.size() / 2],
        s.yawKnotSec,
        longer(40.0),
        longer(60.0),
        longer(120.0),
        s.yawKnotRateStd,
        s.yawKnotRateRwStd,
        s.yawKnotShiftStd,
        prob_.yawKnots_numTooLong());
  }

  if (s.gravityScale != 1.0) {
    auto& gravity = prob_.gravityWorld().value;
    gravity.radius *= s.gravityScale;
    gravity.vec *= s.gravityScale;
    if (verbosity_ != Muted) {
      XR_LOGI("Gravity magnitude set to {}", gravity.radius);
    }
  }
  if (s.gravityKnotSec > 0.0) {
    const auto sortedRigs = prob_.sortedRigIndices();
    const int64_t startUs = prob_.inertialPose(sortedRigs.front()).timestampUs;
    const int64_t endUs = prob_.inertialPose(sortedRigs.back()).timestampUs;
    const int numKnots =
        std::max(2, int(std::ceil((endUs - startUs) * 1e-6 / s.gravityKnotSec)) + 1);
    // world z is the vertical; a non-positive std holds the component (1e-6 m/s2)
    const auto sqrtH = [](double stdDev) { return stdDev > 0.0 ? 1.0 / stdDev : 1e6; };
    prob_.gravityKnots_init(
        startUs,
        endUs,
        numKnots,
        Vec3(sqrtH(s.gravityDirStd), sqrtH(s.gravityDirStd), sqrtH(s.gravityMagStd)),
        Vec3(sqrtH(s.gravityRwDirStd), sqrtH(s.gravityRwDirStd), sqrtH(s.gravityRwMagStd)));
    if (verbosity_ != Muted) {
      XR_LOGI(
          "Gravity as a function of time: {} knots, one every {:.1f} s; std of magnitude / "
          "direction: prior {} / {}, random walk {} / {} m/s2",
          numKnots,
          (endUs - startUs) * 1e-6 / (numKnots - 1),
          s.gravityMagStd,
          s.gravityDirStd,
          s.gravityRwMagStd,
          s.gravityRwDirStd);
    }
  }

  if (s.earthRateUp != 0.0) {
    // world z is the vertical
    prob_.setEarthRateWorld(Vec3(0.0, 0.0, s.earthRateUp));
    if (verbosity_ != Muted) {
      XR_LOGI("Rotation of the Earth about the vertical: {} rad/s", s.earthRateUp);
    }
  }
  if (s.scaleKnotSec > 0.0) {
    const auto sortedRigs = prob_.sortedRigIndices();
    const int64_t startUs = prob_.inertialPose(sortedRigs.front()).timestampUs;
    const int64_t endUs = prob_.inertialPose(sortedRigs.back()).timestampUs;
    const int numKnots = std::max(2, int(std::ceil((endUs - startUs) * 1e-6 / s.scaleKnotSec)) + 1);
    prob_.scaleKnots_init(startUs, endUs, numKnots, 1.0 / s.scaleKnotStd, 1.0 / s.scaleKnotRwStd);
    if (verbosity_ != Muted) {
      XR_LOGI(
          "Scale of the IMU's metric as a function of time: {} knots, one every {:.1f} s; prior "
          "std {}, random walk std {}",
          numKnots,
          (endUs - startUs) * 1e-6 / (numKnots - 1),
          s.scaleKnotStd,
          s.scaleKnotRwStd);
    }
  }
  // required prior to adding inertial factors and omega priors
  regenerateAllPreintegrationsFromImuMeasurements();

  addInertialFactors(s.imuLossRadius, s.imuLossCutoff);

  if (verbosity_ != Muted) {
    XR_LOGI("Adding all random walk factors");
  }
  addAllRandomWalkFactors(
      s.imuRWinflate, s.camIntrRWinflate, s.imuExtrRWinflate, s.camExtrRWinflate);

  if (verbosity_ != Muted) {
    XR_LOGI("Adding omega prior factors");
  }
  addOmegaPriors();

  // factory calibration priors
  if (s.imuFactorCalibInflate > 0.0) {
    addImuFactoryCalibPriors(s.imuFactorCalibInflate);
  }

  if (s.camIntrFactorCalibInflate > 0.0) {
    addCamIntrFactoryCalibPriors(s.camIntrFactorCalibInflate);
  }

  if (s.imuExtrFactorCalibInflate > 0.0) {
    addImuExtrFactoryCalibPriors(s.imuExtrFactorCalibInflate);
  }

  if (s.camExtrFactorCalibInflate > 0.0) {
    addCamExtrFactoryCalibPriors(s.camExtrFactorCalibInflate);
  }
}

void SingleSessionAdapter::initAllParams(const ParamInitSettings& settings) {
  // validate settings:
  constexpr int kMinNumberRigs = 5;
  const int64_t rigIndexStart = settings.rigIndexStart >= 0 ? settings.rigIndexStart : 0;
  const int64_t rigIndexEnd =
      settings.rigIndexEnd >= 0 ? settings.rigIndexEnd : numRigsInRecording_;

  XR_CHECK_GE(
      rigIndexEnd - rigIndexStart,
      kMinNumberRigs,
      "Too small problem size, requested rig range {}..{}",
      rigIndexStart,
      rigIndexEnd);

  prob_.gravityWorld_initNew({
      .radius = kDefaultGravityMagnitude,
      .vec = Vec3{0, 0, -kDefaultGravityMagnitude},
  });

  if (verbosity_ != Muted) {
    XR_LOGI("Loading rigs...");
  }
  initRigs(rigIndexStart, rigIndexEnd);

  if (settings.poseConstant || settings.velConstant || settings.omegaConstant) {
    prob_.setAllRigsConstant(settings.poseConstant, settings.velConstant, settings.omegaConstant);
  }

  initNonRigParams(settings);
}

void SingleSessionAdapter::initAllParamsInterpolatingRigPoses(
    const ParamInitSettings& settings,
    const std::vector<TimeStampInterval>& timeIntervals,
    const std::vector<int64_t>& krTimestampsUs,
    const std::vector<Sophus::SE3d>& Ts_kr_world,
    const TrajectoryBase* trajectory,
    const std::shared_ptr<GravityVariable>& gravityWorld,
    StatsValueContainer* frameDistortionStats_relRot,
    StatsValueContainer* frameDistortionStats_relTr,
    int rigWindowGrow) {
  prob_.gravityWorld_setRef(gravityWorld);

  if (verbosity_ != Muted) {
    XR_LOGI("Loading rigs with map priors...");
  }
  initRigsInterpolatingPoses(
      timeIntervals,
      krTimestampsUs,
      Ts_kr_world,
      trajectory,
      settings.velToGt,
      settings.omegaToGt,
      frameDistortionStats_relRot,
      frameDistortionStats_relTr,
      rigWindowGrow);

  if (settings.poseConstant || settings.velConstant || settings.omegaConstant) {
    prob_.setAllRigsConstant(settings.poseConstant, settings.velConstant, settings.omegaConstant);
  }

  initNonRigParams(settings);
}

void SingleSessionAdapter::initAllParamsWithGtTrajectory(
    const ParamInitSettings& settings,
    const std::vector<TimeStampInterval>& timeIntervals,
    const std::vector<int64_t>& krTimestampsUs,
    const TrajectoryBase* trajectory,
    const std::shared_ptr<GravityVariable>& gravityWorld,
    int rigWindowGrow) {
  prob_.gravityWorld_setRef(gravityWorld);

  if (verbosity_ != Muted) {
    XR_LOGI("Loading rigs with gt poses...");
  }
  initRigsFromGtTrajectory(
      timeIntervals,
      krTimestampsUs,
      trajectory,
      settings.poseToGt,
      settings.velToGt,
      settings.omegaToGt,
      rigWindowGrow);

  if (settings.poseConstant || settings.velConstant || settings.omegaConstant) {
    prob_.setAllRigsConstant(settings.poseConstant, settings.velConstant, settings.omegaConstant);
  }

  initNonRigParams(settings);
}

void SingleSessionAdapter::initAllParamsWithGtTrajectory(
    const ParamInitSettings& settings,
    const TrajectoryBase* trajectory) {
  // validate settings:
  constexpr int kMinNumberRigs = 5;
  const int64_t rigIndexStart = settings.rigIndexStart >= 0 ? settings.rigIndexStart : 0;
  const int64_t rigIndexEnd =
      settings.rigIndexEnd >= 0 ? settings.rigIndexEnd : numRigsInRecording_;

  XR_CHECK_GE(
      rigIndexEnd - rigIndexStart,
      kMinNumberRigs,
      "Too small problem size, requested rig range {}..{}",
      rigIndexStart,
      rigIndexEnd);

  // gravity from GT trajectory if and only if we use GT poses
  prob_.gravityWorld_initNew({
      .radius = kDefaultGravityMagnitude,
      .vec = settings.poseToGt ? trajectory->gravity() : Vec3{0, 0, -kDefaultGravityMagnitude},
  });

  if (verbosity_ != Muted) {
    XR_LOGI("Loading all rigs with gt poses...");
  }
  initRigsFromGtTrajectory(
      rigIndexStart,
      rigIndexEnd,
      trajectory,
      settings.poseToGt,
      settings.velToGt,
      settings.omegaToGt);

  if (settings.poseConstant || settings.velConstant || settings.omegaConstant) {
    prob_.setAllRigsConstant(settings.poseConstant, settings.velConstant, settings.omegaConstant);
  }

  initNonRigParams(settings);
}

int64_t SingleSessionAdapter::usedRecordingLengthUs() const {
  auto rigs = prob_.sortedRigIndices();
  const auto& iPoseStart = fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[rigs[0]]];
  const auto& iPoseEnd = fData_.inertialPoses[matcher_.rigIndexToEvolvingStateIndex[rigs.back()]];
  return iPoseEnd.timestamp_us - iPoseStart.timestamp_us;
}

SingleSessionProblem& SingleSessionAdapter::getProblem() {
  return prob_;
}

const SingleSessionProblem& SingleSessionAdapter::getProblem() const {
  return prob_;
}

} // namespace visual_inertial_ba
