/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/loop_closure/DriftMapping.h>

#include <Eigen/Dense>

#include <stdexcept>

namespace visual_inertial_ba::loop {

DriftCorrection fitDriftCorrection(
    const std::vector<int64_t>& rigTimesUs,
    const std::vector<Vec3>& rigPos,
    const std::vector<int>& nodeRig,
    const std::vector<double>& nodeYaw,
    const std::vector<Vec3>& nodePos) {
  if (rigTimesUs.size() < 2 || rigTimesUs.size() != rigPos.size() || nodeRig.empty() ||
      nodeYaw.size() != nodeRig.size() || nodePos.size() != nodeRig.size()) {
    throw std::invalid_argument("fitDriftCorrection: inconsistent input");
  }
  DriftCorrection c;
  YawDriftPath& path = c.path;
  path.active = true;
  path.shift = true;
  for (const int64_t t : rigTimesUs) {
    path.timeSec.push_back((t - rigTimesUs.front()) * 1e-6);
  }
  path.refPos = rigPos;
  // knot times: the start, the nodes, the end
  auto& times = path.knotTimeSec;
  times.push_back(0.0);
  std::vector<int> nodeInterval(nodeRig.size(), -1); // the interval that starts at the node
  for (size_t k = 0; k < nodeRig.size(); k++) {
    if (k > 0 && nodeRig[k] <= nodeRig[k - 1]) {
      throw std::invalid_argument("fitDriftCorrection: nodes not in time order");
    }
    const double t = path.timeSec[nodeRig[k]];
    if (t > times.back()) {
      times.push_back(t);
    }
    nodeInterval[k] = int(times.size()) - 1;
  }
  if (path.timeSec.back() > times.back()) {
    times.push_back(path.timeSec.back());
  }
  if (times.size() < 2) {
    times.push_back(path.timeSec.back() + 1.0);
  }
  c.knots.assign(times.size() - 1, Vec4::Zero());
  // rates
  for (size_t k = 0; k + 1 < nodeRig.size(); k++) {
    const double dt = path.timeSec[nodeRig[k + 1]] - path.timeSec[nodeRig[k]];
    c.knots[nodeInterval[k]][0] = (nodeYaw[k + 1] - nodeYaw[k]) / dt;
  }
  const auto rate = [&](int k) { return c.knots[k][0]; };
  std::vector<double> psi(path.timeSec.size());
  for (size_t i = 0; i < psi.size(); i++) {
    psi[i] = path.psiAt(path.timeSec[i], rate);
  }
  // drift velocities, node after node; up to the first node the knots are 0, so q = ref there
  Vec3 q = rigPos[nodeRig[0]];
  for (size_t k = 0; k + 1 < nodeRig.size(); k++) {
    Vec3 base = q;
    Mat33 M = Mat33::Zero();
    for (int j = nodeRig[k]; j < nodeRig[k + 1]; j++) {
      const Mat33 R = SO3::rotZ(0.5 * (psi[j] + psi[j + 1])).matrix();
      base += R * (rigPos[j + 1] - rigPos[j]);
      M += (path.timeSec[j + 1] - path.timeSec[j]) * R;
    }
    const Vec3 u = M.fullPivLu().solve(nodePos[k + 1] - base);
    c.knots[nodeInterval[k]].tail<3>() = u;
    q = base + M * u;
  }
  return c;
}

void driftStates(const DriftCorrection& c, std::vector<double>& psi, std::vector<Vec3>& q) {
  const YawDriftPath& path = c.path;
  const auto rate = [&](int k) { return c.knots[k][0]; };
  const auto vel = [&](int k) { return Vec3(c.knots[k].tail<3>()); };
  const size_t n = path.timeSec.size();
  psi.resize(n);
  q.resize(n);
  for (size_t i = 0; i < n; i++) {
    psi[i] = path.psiAt(path.timeSec[i], rate);
    q[i] = i == 0 ? path.refPos[0]
                  : Vec3(q[i - 1] +
                         SO3::rotZ(0.5 * (psi[i - 1] + psi[i])) *
                             (path.refPos[i] - path.refPos[i - 1] +
                              path.shiftBetween(path.timeSec[i - 1], path.timeSec[i], vel)));
  }
}

} // namespace visual_inertial_ba::loop
