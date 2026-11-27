/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <viba/problem/Types.h>

#include <cstdint>
#include <vector>

// Calibrated two-view geometry on bearing vectors (unit rays in the camera frames).
//
// Convention: a point X1 in the frame of camera 1 is X2 = R X1 + t in the frame of camera 2;
// E = [t]x R, so that f2^T E f1 = 0 for the bearings f1, f2 of one point. The scale of t is not
// observable: |t| = 1. The centre of camera 2 seen from camera 1 is -R^T t.
namespace visual_inertial_ba::loop {

// The essential matrices (|E| = 1, up to 10) consistent with five bearing pairs. Method: the
// 4-dimensional null space of the 5 x 9 epipolar system, E = x X + y Y + z Z + W; the ten cubic
// constraints det E = 0 and 2 E E^T E - tr(E E^T) E = 0 in (x, y, z); elimination of the ten cubic
// monomials (a 10 x 10 linear solve), so that every cubic is a combination of the ten monomials of
// degree <= 2; the eigenvectors of the 10 x 10 matrix of the multiplication by x on those ten
// monomials give the solutions (real eigenvalues only).
std::vector<Mat33> fivePointEssential(const Vec3* f1, const Vec3* f2);

// Relative pose: X2 = R X1 + t, |t| = 1
struct RelativePose {
  Mat33 R = Mat33::Identity();
  Vec3 t = Vec3::UnitX();
  // direction of the centre of camera 2 seen from camera 1 (in camera 1)
  Vec3 directionInCamera1() const {
    return -(R.transpose() * t);
  }
};

// angular epipolar error of a bearing pair: the mean of the sines of the angles of each ray to the
// epipolar plane of the other one (radians for small values)
double epipolarAngularError(const Mat33& E, const Vec3& f1, const Vec3& f2);

// The four decompositions of E; the one with the most points in front of both cameras among the
// given pairs (indices into f1, f2). Returns that number (0: none). With a rotation prior, only the
// rotation nearer to it is considered (the sign of t still by cheirality).
int decomposeEssential(
    const Mat33& E,
    const std::vector<Vec3>& f1,
    const std::vector<Vec3>& f2,
    const std::vector<int>& indices,
    RelativePose& pose,
    const Mat33* rotationPrior = nullptr);

// depths along the two rays (least squares, may be negative) of the point seen as f1, f2
void triangulateDepths(const RelativePose& pose, const Vec3& f1, const Vec3& f2, double& d1, double& d2);

struct RansacSettings {
  int maxIterations = 1000;
  double thresholdRad = 0.004; // angular epipolar error of an inlier
  double confidence = 0.9999;
  uint32_t seed = 1;
  int refineIterations = 50; // Levenberg-Marquardt on (R, direction of t) with all inliers
  // Optional prior on R (e.g. from the open-loop poses): of the two rotations of the decomposition
  // of E, the one nearer to it is taken (cheirality cannot tell them apart when the parallax is
  // small), and the result is refused if R is farther than maxPriorAngleRad from it.
  bool useRotationPrior = false;
  Mat33 rotationPrior = Mat33::Identity();
  double maxPriorAngleRad = 0.35;
  // With the rotation prior, a prior translation direction (X2 = R X1 + t, unit): the refinement
  // is also started from (rotationPrior, translationPrior) and the result with the lower cost is
  // kept. Two-view geometry with little parallax has separate local minima (the synthetic test
  // found one 0.5 deg away from the truth, at a higher cost).
  bool useTranslationPrior = false;
  Vec3 translationPrior = Vec3::UnitX();
};

struct RansacResult {
  bool ok = false;
  RelativePose pose;
  std::vector<int> inliers; // indices of the inlier pairs
  double medianParallaxRad = 0.0; // median angle between the two rays of the inliers
  double rmsErrorRad = 0.0; // rms angular error of the inliers after refinement
  // standard deviations from the refinement (rms error^2 (J^T J)^-1, largest eigenvalue): of the
  // rotation, and of the direction of t [rad]; 0 if not computed
  double sigmaRotationRad = 0.0;
  double sigmaDirectionRad = 0.0;
};

// 5-point RANSAC, cheirality, then a refinement of R and the direction of t on the inliers
RansacResult ransacRelativePose(
    const std::vector<Vec3>& f1,
    const std::vector<Vec3>& f2,
    const RansacSettings& settings);

} // namespace visual_inertial_ba::loop
