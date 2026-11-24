/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/common/Enumerate.h>
#include <viba/single_session/SingleSessionAdapter.h>

#define DEFAULT_LOG_CHANNEL "ViBa::VisualFactors"
#include <logging/Log.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace visual_inertial_ba {

void SingleSessionAdapter::setPointHosts(const std::string& pointParam, double pointFarDist) {
  XR_CHECK(
      pointParam == "world" || pointParam == "host" || pointParam == "hostinv" ||
          pointParam == "hostmix",
      "Unknown point parameterization '{}'",
      pointParam);
  if (pointParam == "world") {
    return;
  }

  int64_t nInverseDepth = 0;
  std::vector<std::pair<int64_t, int64_t>> timeRig;
  for (size_t ptVarIndex = 0; ptVarIndex < prob_.numPointTrackVariables(); ptVarIndex++) {
    // host: the rig of the middle observation in time
    timeRig.clear();
    for (const auto& obsIndex : pointTrackObservations_[ptVarIndex]) {
      const auto& procObs = fData_.trackingObservations[obsIndex];
      timeRig.emplace_back(
          procObs.captureTimestampUs,
          findOrDie(matcher_.timestampToRigIndex, procObs.captureTimestampUs));
    }
    std::sort(timeRig.begin(), timeRig.end());
    const int64_t hostRig = timeRig[timeRig.size() / 2].second;

    bool inverseDepth = pointParam == "hostinv";
    if (pointParam == "hostmix") {
      const Vec3 ptHost =
          prob_.inertialPose(hostRig).T_bodyImu_world.value * prob_.pointTrack(ptVarIndex).value;
      inverseDepth = ptHost.norm() > pointFarDist;
    }
    prob_.pointTrack_setHost(ptVarIndex, hostRig, inverseDepth);
    nInverseDepth += inverseDepth;
  }
  if (verbosity_ != Muted) {
    XR_LOGI(
        "Points expressed in a host rig ({}): {} as inverse depth, {} as position",
        pointParam,
        nInverseDepth,
        prob_.numPointTrackVariables() - nInverseDepth);
  }
}

void SingleSessionAdapter::addVisualFactors(
    double trackingObsLossRadius,
    double trackingObsLossCutoff,
    bool optimizeDetectorBias,
    const std::string& pointParam,
    double pointFarDist) {
  // reset loss radius
  prob_.reprojErrorLoss().setSize(trackingObsLossRadius, trackingObsLossCutoff);

  if (optimizeDetectorBias) {
    throw std::runtime_error("Detector bias currently unsupported");
    // prob_.detectorBiases_init(numCameras_);
  }

  setPointHosts(pointParam, pointFarDist);

  int64_t nAdded = 0;
  for (size_t ptVarIndex = 0; ptVarIndex < prob_.numPointTrackVariables(); ptVarIndex++) {
    for (const auto& obsIndex : pointTrackObservations_[ptVarIndex]) {
      const auto& procObs = fData_.trackingObservations[obsIndex];
      const int64_t rigIndex = findOrDie(matcher_.timestampToRigIndex, procObs.captureTimestampUs);
      const int64_t cameraIndex = procObs.cameraIndex;
      auto& pointTrackVar = prob_.pointTrack(ptVarIndex);
      Vec2 projBaseRes = procObs.projectionBaseRes.cast<double>();
      Mat22 sqrtH_BaseRes = procObs.sqrtH_BaseRes.cast<double>();

      if (optimizeDetectorBias) {
        prob_.addVisualFactorWithBias(
            rigIndex,
            cameraIndex,
            pointTrackVar,
            prob_.detectorBias(cameraIndex),
            projBaseRes,
            sqrtH_BaseRes,
            prob_.reprojErrorLoss());
      } else {
        prob_.addVisualFactor(
            rigIndex,
            cameraIndex,
            static_cast<int64_t>(ptVarIndex),
            projBaseRes,
            sqrtH_BaseRes,
            prob_.reprojErrorLoss());
      }
      nAdded++;
    }
  }
  if (verbosity_ != Muted) {
    XR_LOGI("Added (tracking) visual factors: {}", nAdded);
  }
}

} // namespace visual_inertial_ba
