/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/single_session/SingleSessionAdapter.h>
#include <viba/single_session/Triangulation.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

#define DEFAULT_LOG_CHANNEL "ViBa::InitPointTracks"
#include <logging/Log.h>

namespace visual_inertial_ba {

// filter observations referencing rigs which will be optimized
void SingleSessionAdapter::filterPointObservations(
    std::vector<int64_t>& filteredObsIndices,
    const std::vector<int64_t>& obsIndices) {
  for (auto obsIndex : obsIndices) {
    const auto& procObs = fData_.trackingObservations[obsIndex];
    const int64_t rigIndex = findOrDie(matcher_.timestampToRigIndex, procObs.captureTimestampUs);
    if (prob_.inertialPose_exists(rigIndex)) {
      filteredObsIndices.push_back(obsIndex);
    }
  }
}

SingleSessionAdapter::SkyMeasure SingleSessionAdapter::skyMeasure(
    const Vec3& pointWorld,
    const std::vector<int64_t>& obsIndices) const {
  const Vec3 up = -prob_.gravityWorld().value.vec.normalized();
  SkyMeasure m{
      .minElevationRad = std::numeric_limits<double>::infinity(),
      .minDist = std::numeric_limits<double>::infinity(),
      .baseline = 0.0,
  };
  Vec3 firstCenter = Vec3::Zero();
  bool first = true;
  for (const auto obsIndex : obsIndices) {
    const auto& procObs = fData_.trackingObservations[obsIndex];
    const int64_t rigIndex = findOrDie(matcher_.timestampToRigIndex, procObs.captureTimestampUs);
    const SE3 T_cam_world = prob_.T_Cam_BodyImu(rigIndex, procObs.cameraIndex).var.value *
        prob_.inertialPose(rigIndex).T_bodyImu_world.value;
    const Vec3 center = T_cam_world.inverse().translation();
    const Vec3 toPoint = pointWorld - center;
    const double dist = toPoint.norm();
    m.minDist = std::min(m.minDist, dist);
    m.minElevationRad =
        std::min(m.minElevationRad, std::asin(std::clamp(toPoint.dot(up) / dist, -1.0, 1.0)));
    if (first) {
      firstCenter = center;
      first = false;
    } else {
      m.baseline = std::max(m.baseline, (center - firstCenter).norm());
    }
  }
  return m;
}

void SingleSessionAdapter::initPointsFromObservations() {
  int64_t usedObs = 0, totalObs = 0, triedTracks = 0, skippedTracks = 0;
  // sky points: counts by distance and by distance over baseline, for the log
  const bool skyFilter = skyMinDist_ > 0.0 || skyMinDistOverBaseline_ > 0.0;
  const double skyMinElevationRad = degreesToRadians(skyMinElevationDeg_);
  constexpr std::array<double, 5> kSkyDists = {30.0, 100.0, 300.0, 1000.0, 10000.0};
  constexpr std::array<double, 5> kSkyRatios = {30.0, 100.0, 300.0, 1000.0, 10000.0};
  std::array<int64_t, 5> aboveByDist{}, aboveByRatio{}, belowByDist{}, belowByRatio{};
  int64_t aboveTracks = 0, skyTracks = 0, skyObs = 0;
  std::vector<int64_t> filteredObsIndices;
  for (auto [ptId, ptIndex] : matcher_.pointIdToPointIndex) {
    XR_CHECK_LT(ptIndex, matcher_.pointIndexToObsIndices.size());
    filterPointObservations(filteredObsIndices, matcher_.pointIndexToObsIndices[ptIndex]);
    if (filteredObsIndices.size() < triangulation::kMinInlierObs) {
      skippedTracks += !filteredObsIndices.empty(); // some observations, but still too short track
      filteredObsIndices.clear();
      continue;
    }

    triedTracks++;
    totalObs += filteredObsIndices.size();

    constexpr int kSeedOffset = 1729; // to map point ids to ransac seeds
    auto maybeTriangulationResult = triangulatePoint(filteredObsIndices, ptId + kSeedOffset);
    if (maybeTriangulationResult) {
      const SkyMeasure sky = skyMeasure(
          maybeTriangulationResult->point, maybeTriangulationResult->inlierObservationIndices);
      const bool above = sky.minElevationRad >= skyMinElevationRad;
      aboveTracks += above;
      for (size_t i = 0; i < kSkyDists.size(); i++) {
        (above ? aboveByDist : belowByDist)[i] += sky.minDist > kSkyDists[i];
        (above ? aboveByRatio : belowByRatio)[i] += sky.minDist > kSkyRatios[i] * sky.baseline;
      }
      if (above &&
          ((skyMinDist_ > 0.0 && sky.minDist > skyMinDist_) ||
           (skyMinDistOverBaseline_ > 0.0 &&
            sky.minDist > skyMinDistOverBaseline_ * sky.baseline))) {
        skyTracks++;
        skyObs += maybeTriangulationResult->inlierObservationIndices.size();
        filteredObsIndices.clear();
        continue;
      }
      usedObs += maybeTriangulationResult->inlierObservationIndices.size();
      int64_t varIndex = prob_.pointTrack_addNew(maybeTriangulationResult->point);
      XR_CHECK_EQ(varIndex, pointTrackObservations_.size());
      pointTrackObservations_.push_back(
          std::move(maybeTriangulationResult->inlierObservationIndices));
    }

    filteredObsIndices.clear();
  }
  if (verbosity_ != Muted) {
    XR_LOGI(
        "Created {} point tracks out of {} attemped ({} skipped / small overlap). Will use {}/{} observations",
        prob_.numPointTrackVariables(),
        triedTracks,
        skippedTracks,
        usedObs,
        totalObs);
    XR_LOGI(
        "Triangulated tracks at least {} deg above the horizon from all their cameras: {}; of "
        "these, farther than 30 / 100 / 300 / 1000 / 10000 m: {} / {} / {} / {} / {}, farther "
        "than 30 / 100 / 300 / 1000 / 10000 baselines: {} / {} / {} / {} / {}",
        skyMinElevationDeg_,
        aboveTracks,
        aboveByDist[0], aboveByDist[1], aboveByDist[2], aboveByDist[3], aboveByDist[4],
        aboveByRatio[0], aboveByRatio[1], aboveByRatio[2], aboveByRatio[3], aboveByRatio[4]);
    XR_LOGI(
        "Other triangulated tracks: farther than 30 / 100 / 300 / 1000 / 10000 m: {} / {} / {} "
        "/ {} / {}, farther than 30 / 100 / 300 / 1000 / 10000 baselines: {} / {} / {} / {} / {}",
        belowByDist[0], belowByDist[1], belowByDist[2], belowByDist[3], belowByDist[4],
        belowByRatio[0], belowByRatio[1], belowByRatio[2], belowByRatio[3], belowByRatio[4]);
    if (skyFilter) {
      XR_LOGI("Left out as sky points: {} tracks, {} observations", skyTracks, skyObs);
    }
  }
}

} // namespace visual_inertial_ba
