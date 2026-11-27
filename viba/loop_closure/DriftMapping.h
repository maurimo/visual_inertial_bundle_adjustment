/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <viba/problem/Types.h>
#include <viba/problem/YawDrift.h>

#include <vector>

// The corrections of the pose graph at its nodes (yaw correction, new position) put into the whole
// trajectory with the transform of VI-BA's drift knots (YawDrift.h; the same formulas as
// SingleSessionProblem::yawKnots_trueStates): knot intervals = the intervals between the nodes;
// on each, a constant yaw rate w and a constant drift velocity u. With the variables at the frozen
// copy (the open-loop trajectory), the transform gives every rig
//   R_world_rig -> Rz(Psi_i) R_world_rig,  vel -> Rz(Psi_i) vel,  p -> q_i,
//   q_{i+1} = q_i + Rz((Psi_i + Psi_{i+1}) / 2) (ref_{i+1} - ref_i + U_i)
// so each step of the trajectory is turned by the yaw accumulated up to it and stretched by the
// drift velocity: the local shape is kept, as for a gyro-bias-like drift.
//
// The knot values are fitted exactly: w_k = (yaw_{k+1} - yaw_k) / (t_{k+1} - t_k) reproduces the
// nodes' yaw corrections; u_k solves the 3 x 3 linear equation that puts node k + 1 at its new
// position given node k. Before the first node and after the last one the knots are 0: the first
// node must keep its pose (the pose graph holds it).
namespace visual_inertial_ba::loop {

struct DriftCorrection {
  YawDriftPath path; // timeSec / refPos of all rigs, knotTimeSec
  std::vector<Vec4> knots; // (w [rad/s], u [m/s]) per interval
};

// rigTimesUs, rigPos: all rigs in time order (IMU positions of the open-loop trajectory);
// nodeRig: indices into them of the nodes, increasing; nodeYaw / nodePos: the nodes' yaw
// corrections and new positions (nodeYaw[0] = 0, nodePos[0] = rigPos[nodeRig[0]])
DriftCorrection fitDriftCorrection(
    const std::vector<int64_t>& rigTimesUs,
    const std::vector<Vec3>& rigPos,
    const std::vector<int>& nodeRig,
    const std::vector<double>& nodeYaw,
    const std::vector<Vec3>& nodePos);

// Psi_i and q_i of every rig (the new rotation is Rz(Psi_i) R_world_rig, the new position q_i)
void driftStates(const DriftCorrection& c, std::vector<double>& psi, std::vector<Vec3>& q);

} // namespace visual_inertial_ba::loop
