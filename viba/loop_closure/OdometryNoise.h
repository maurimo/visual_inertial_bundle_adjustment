/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <viba/problem/Types.h>

#include <algorithm>
#include <cmath>

namespace visual_inertial_ba::loop {

// What the tracking was like between two loop-closing key rigs (see IntervalStats in
// LoopClosure.h for how it is measured): the number of point tracks observed in the interval and
// their mean length.
struct IntervalStats {
  double numTracks = 0.0;
  double meanTrackSec = 0.0;
};

// Uncertainty of the open-loop motion between two loop-closing key rigs (the odometry edges of
// both pose-graph stages), in the 4 degrees of freedom of the pose graph: yaw, and the position
// difference expressed in the world turned by the yaw of the first rig.
//
// Model: a random walk in time and travelled distance, its variance scaled by the tracking
// quality of the interval:
//   var_yaw    = q * (yawPerSqrtSec^2 dt)                                    (+ floor^2)
//   var_pos_h  = q * (posHPerSqrtSec^2 dt + posPerSqrtMeter^2 dist)          (+ floor^2)
//   var_pos_v  = q * (posVPerSqrtSec^2 dt + posPerSqrtMeter^2 dist)          (+ floor^2)
//   q = clamp((refNumTracks / numTracks)^tracksExponent, 1/maxQuality, maxQuality)
//     * clamp((refTrackSec / meanTrackSec)^trackLengthExponent, 1/maxQuality, maxQuality)
//
// THE PARAMETERS ARE NOT FITTED. The defaults are placeholders of the right order of magnitude
// (VI-BA's open-loop heading drift on R_12_10cp is about 1-2 deg over 1000 s). They are set here
// and can be overridden on the command line of ark_vi_ba (--lc-odo-*). How to fit them: LC_IMPL.md
// in the main repository ("Odometry uncertainty: fitting").
struct OdometryNoiseModel {
  double yawPerSqrtSec = 3e-4; // rad / sqrt(s)
  double posHPerSqrtSec = 5e-3; // m / sqrt(s), each horizontal axis
  double posVPerSqrtSec = 3e-3; // m / sqrt(s)
  double posPerSqrtMeter = 3e-3; // m / sqrt(m) of travelled distance, every axis
  double yawFloor = 1e-4; // rad
  double posFloor = 1e-3; // m
  double refNumTracks = 1000.0;
  double tracksExponent = 1.0; // 1: variance inversely proportional to the number of tracks
  double refTrackSec = 3.0;
  double trackLengthExponent = 1.0; // 1: variance inversely proportional to the track length
  double maxQuality = 10.0;

  struct Sigmas {
    double yaw = 0.0;
    Vec3 pos = Vec3::Zero();
  };

  double quality(const IntervalStats& s) const {
    auto ratio = [&](double ref, double value, double exponent) {
      if (exponent == 0.0) {
        return 1.0;
      }
      const double r = std::pow(ref / std::max(value, 1e-9), exponent);
      return std::clamp(r, 1.0 / maxQuality, maxQuality);
    };
    return ratio(refNumTracks, s.numTracks, tracksExponent) *
        ratio(refTrackSec, s.meanTrackSec, trackLengthExponent);
  }

  Sigmas sigmas(double dtSec, double distM, const IntervalStats& s) const {
    const double q = quality(s);
    dtSec = std::max(dtSec, 0.0);
    distM = std::max(distM, 0.0);
    const double d2 = posPerSqrtMeter * posPerSqrtMeter * distM;
    Sigmas out;
    out.yaw = std::sqrt(q * yawPerSqrtSec * yawPerSqrtSec * dtSec + yawFloor * yawFloor);
    const double h = std::sqrt(q * (posHPerSqrtSec * posHPerSqrtSec * dtSec + d2) + posFloor * posFloor);
    const double v = std::sqrt(q * (posVPerSqrtSec * posVPerSqrtSec * dtSec + d2) + posFloor * posFloor);
    out.pos = Vec3(h, h, v);
    return out;
  }
};

} // namespace visual_inertial_ba::loop
