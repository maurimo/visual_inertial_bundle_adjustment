/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <viba/loop_closure/DriftMapping.h>
#include <viba/loop_closure/FivePoint.h>
#include <viba/loop_closure/MatchSource.h>
#include <viba/loop_closure/OdometryNoise.h>
#include <viba/loop_closure/PoseGraph.h>

#include <functional>
#include <ostream>
#include <string>
#include <vector>

// Loop closure after the open-loop stages of VI-BA (see LC_IMPL.md in the main repository):
//   1. loop-closing key rigs (LC-KRs): one key rig with matches every lckrSpacingSec (the "grid");
//      for each, pose-prior candidates among the earlier key rigs (within candidateRadiusM, more
//      than candidateMinGapSec earlier, nearest first, at most maxPartners at least
//      partnerSeparationSec apart), verified by 5-point RANSAC on the correspondences of the match
//      source (the camera combination with most inliers; at least minInliers). The nodes are the
//      grid rigs and the partners of the verified pairs.
//   2. stage 1: 4-DoF pose graph, odometry edges between consecutive nodes + one loop edge per
//      pair (relative yaw, direction of the relative position; the scale of the 5-point
//      translation is not used);
//   3. stage 2: odometry edges + the inlier matches of the pairs as points with reprojection
//      errors in both rigs, started from stage 1;
//   both stages with GNC over the pairs, and in one of two variants (PoseGraphVariant);
//   4. the corrections put into every key rig through the drift-knot transform (DriftMapping.h).
namespace visual_inertial_ba::loop {

struct KeyRig {
  int64_t rigIndex = -1;
  int64_t timestampUs = 0;
  SE3 T_world_body; // open-loop pose (true states)
};

// cameras (models and extrinsics) at a key rig
using CamerasAtRig = std::function<std::vector<CameraAtRig>(int64_t rigIndex)>;
// tracking statistics of the open-loop problem in [t0, t1]
using IntervalStatsFn = std::function<IntervalStats(int64_t t0Us, int64_t t1Us)>;

struct LoopClosureSettings {
  PoseGraphVariant variant = PoseGraphVariant::RigPoses;
  // LC-KRs and candidates
  double lckrSpacingSec = 20.0;
  double candidateRadiusM = 25.0;
  double candidateMinGapSec = 30.0;
  int maxPartners = 2;
  double partnerSeparationSec = 8.0;
  int64_t timeToleranceUs = 2000; // match-source times against key-rig times
  // verification
  int minMatches = 20; // correspondences of a camera combination before RANSAC
  int minInliers = 20;
  RansacSettings ransac;
  // the relative rotation of the open-loop poses as a prior of the 5-point decomposition (and a
  // gate: farther than maxRotationPriorDeg refuses the pair)
  bool useRotationPrior = true;
  double maxRotationPriorDeg = 20.0;
  double minParallaxDeg = 1.0; // below: the pair constrains the rotation only (stage 1)
  // stage 1: the sigmas of a loop edge are sqrt(floor^2 + (fitInflation * sigma of the 5-point
  // refinement)^2), the floors being these two
  double loopYawSigma = 0.005; // rad
  double loopDirSigma = 0.02; // rad
  double fitInflation = 3.0;
  GncSettings gnc1;
  // stage 2
  bool stage2 = true;
  double pixelSigma = 1.0;
  double huberPx = 2.0;
  int maxPointsPerPair = 200;
  GncSettings gnc2;
  // odometry
  OdometryNoiseModel odometry;
  // Knots variant: the knot times [s from the first key rig, numKnots + 1 of them, the last at
  // or after the last key rig]; empty: every knotSec (a rest shorter than knotSec joins the last)
  std::vector<double> knotTimesSec;
  double knotSec = 30.0;
  double knotRateSigma = 0.01; // priors of the knots [rad/s], [m/s]
  double knotVelSigma = 1.0;
  int numThreads = 1;
};

struct LoopClosureResult {
  bool ok = false;
  std::string message;
  std::vector<LcNode> nodes;
  std::vector<int> nodeRig; // node -> index into the key rigs
  std::vector<LoopPair> pairs; // verified
  std::vector<Vec4> stage1, stage2; // node states (yaw correction, position) after each stage
  std::vector<double> weights1, weights2; // final pair weights (stage 2: -1 if not used)
  int numGrid = 0, numCandidates = 0, numVerified = 0, numPoints = 0;
  DriftCorrection correction; // what goes into the trajectory (empty knots: nothing)
};

LoopClosureResult runLoopClosure(
    const std::vector<KeyRig>& rigs,
    const CamerasAtRig& cameras,
    const MatchSource& matches,
    const IntervalStatsFn& stats,
    const LoopClosureSettings& settings,
    std::ostream* log = nullptr);

// knot times every knotSec from 0 to lengthSec (a rest shorter than knotSec joins the last)
std::vector<double> uniformKnotTimes(double lengthSec, double knotSec);

// the corrected pose of every key rig, and the yaw turned (the velocities turn by it too)
void correctedStates(
    const DriftCorrection& correction,
    const std::vector<KeyRig>& rigs,
    std::vector<SE3>& T_world_body,
    std::vector<double>& psi);

} // namespace visual_inertial_ba::loop
