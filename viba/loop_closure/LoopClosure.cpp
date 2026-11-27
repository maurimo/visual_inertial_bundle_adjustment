/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/loop_closure/LoopClosure.h>

#include <viba/loop_closure/Bearing.h>

#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace visual_inertial_ba::loop {

namespace {

// index of the element of the sorted vector nearest to t, within tol (-1: none)
int nearestWithin(const std::vector<int64_t>& sorted, int64_t t, int64_t tol) {
  const auto it = std::lower_bound(sorted.begin(), sorted.end(), t);
  int best = -1;
  int64_t bestDt = tol + 1;
  for (auto c : {it, it == sorted.begin() ? sorted.end() : std::prev(it)}) {
    if (c != sorted.end() && std::llabs(*c - t) < bestDt) {
      bestDt = std::llabs(*c - t);
      best = int(c - sorted.begin());
    }
  }
  return best;
}

} // namespace

std::vector<double> uniformKnotTimes(double lengthSec, double knotSec) {
  std::vector<double> t{0.0};
  if (!(knotSec > 0.0) || lengthSec <= 0.0) {
    t.push_back(std::max(lengthSec, 1e-3));
    return t;
  }
  double end = knotSec;
  while (lengthSec - end >= knotSec) {
    t.push_back(end);
    end += knotSec;
  }
  t.push_back(lengthSec);
  return t;
}

void correctedStates(
    const DriftCorrection& correction,
    const std::vector<KeyRig>& rigs,
    std::vector<SE3>& T_world_body,
    std::vector<double>& psi) {
  T_world_body.resize(rigs.size());
  psi.assign(rigs.size(), 0.0);
  if (correction.knots.empty()) {
    for (size_t i = 0; i < rigs.size(); i++) {
      T_world_body[i] = rigs[i].T_world_body;
    }
    return;
  }
  std::vector<Vec3> q;
  driftStates(correction, psi, q);
  for (size_t i = 0; i < rigs.size(); i++) {
    const SO3 R = SO3::rotZ(psi[i]) * rigs[i].T_world_body.so3();
    T_world_body[i] = SE3(R, q[i]);
  }
}

LoopClosureResult runLoopClosure(
    const std::vector<KeyRig>& rigs,
    const CamerasAtRig& cameras,
    const MatchSource& matchSource,
    const IntervalStatsFn& stats,
    const LoopClosureSettings& s,
    std::ostream* log) {
  LoopClosureResult res;
  auto say = [&](const std::string& msg) {
    if (log) {
      *log << "Loop closure: " << msg << std::endl;
    }
  };
  if (rigs.size() < 2) {
    res.message = "fewer than two key rigs";
    return res;
  }
  for (size_t i = 1; i < rigs.size(); i++) {
    if (rigs[i].timestampUs <= rigs[i - 1].timestampUs) {
      res.message = "key rigs not in time order";
      return res;
    }
  }
  const int nRigs = int(rigs.size());
  auto pos = [&](int i) -> Vec3 { return rigs[i].T_world_body.translation(); };

  // key rigs with images in the match source
  const std::vector<int64_t> srcTimes = matchSource.imageTimesUs();
  std::vector<int64_t> rigSrcTime(nRigs, -1);
  std::map<int64_t, int> srcTimeToRig;
  for (int i = 0; i < nRigs; i++) {
    const int k = nearestWithin(srcTimes, rigs[i].timestampUs, s.timeToleranceUs);
    if (k >= 0) {
      rigSrcTime[i] = srcTimes[k];
      srcTimeToRig[srcTimes[k]] = i;
    }
  }

  // 1. the grid of LC-KRs
  std::vector<int> grid;
  {
    const int64_t t0 = rigs.front().timestampUs, t1 = rigs.back().timestampUs;
    const int64_t spacing = std::llround(s.lckrSpacingSec * 1e6);
    std::vector<int64_t> matchable;
    std::vector<int> matchableRig;
    for (int i = 0; i < nRigs; i++) {
      if (rigSrcTime[i] >= 0) {
        matchable.push_back(rigs[i].timestampUs);
        matchableRig.push_back(i);
      }
    }
    for (int64_t t = t0; spacing > 0 && t <= t1; t += spacing) {
      const int k = nearestWithin(matchable, t, spacing / 2);
      if (k >= 0 && (grid.empty() || grid.back() != matchableRig[k])) {
        grid.push_back(matchableRig[k]);
      }
    }
  }
  res.numGrid = int(grid.size());

  // verification of a candidate pair (rig a earlier than rig b)
  auto verify = [&](int a, int b, LoopPair& out) {
    const auto camsA = cameras(rigs[a].rigIndex), camsB = cameras(rigs[b].rigIndex);
    bool found = false;
    for (const ImageMatches& m : matchSource.matches(rigSrcTime[a], rigSrcTime[b])) {
      if (int(m.pxA.size()) < s.minMatches || m.camA < 0 || m.camB < 0 ||
          m.camA >= int(camsA.size()) || m.camB >= int(camsB.size()) ||
          camsA[m.camA].model == nullptr || camsB[m.camB].model == nullptr) {
        continue;
      }
      std::vector<Vec3> fA, fB;
      std::vector<Vec2> pxA, pxB;
      for (size_t k = 0; k < m.pxA.size(); k++) {
        Vec3 a, b;
        if (bearingFromPixel(*camsA[m.camA].model, m.pxA[k], a) &&
            bearingFromPixel(*camsB[m.camB].model, m.pxB[k], b)) {
          fA.push_back(a);
          fB.push_back(b);
          pxA.push_back(m.pxA[k]);
          pxB.push_back(m.pxB[k]);
        }
      }
      if (int(fA.size()) < s.minMatches) {
        continue;
      }
      // the relative rotation of the two cameras on the open-loop trajectory as the prior
      RansacSettings rs = s.ransac;
      rs.useRotationPrior = s.useRotationPrior;
      rs.maxPriorAngleRad = s.maxRotationPriorDeg * M_PI / 180.0;
      const SE3 T_ba = camsB[m.camB].T_cam_body * rigs[b].T_world_body.inverse() *
          rigs[a].T_world_body * camsA[m.camA].T_cam_body.inverse();
      rs.rotationPrior = T_ba.so3().matrix();
      rs.useTranslationPrior = s.useRotationPrior && T_ba.translation().norm() > 1e-3;
      if (rs.useTranslationPrior) {
        rs.translationPrior = T_ba.translation().normalized();
      }
      const RansacResult r = ransacRelativePose(fA, fB, rs);
      if (!r.ok || int(r.inliers.size()) < s.minInliers ||
          (found && r.inliers.size() <= out.fA.size())) {
        continue;
      }
      found = true;
      out = LoopPair();
      out.camA = m.camA;
      out.camB = m.camB;
      out.pose = r.pose;
      out.medianParallaxRad = r.medianParallaxRad;
      out.directionValid = r.medianParallaxRad >= s.minParallaxDeg * M_PI / 180.0;
      out.sigmaRotationRad = r.sigmaRotationRad;
      out.sigmaDirectionRad = r.sigmaDirectionRad;
      for (const int k : r.inliers) {
        out.pxA.push_back(pxA[k]);
        out.pxB.push_back(pxB[k]);
        out.fA.push_back(fA[k]);
        out.fB.push_back(fB[k]);
      }
    }
    return found;
  };

  // candidates and verified pairs
  std::vector<std::pair<int, int>> pairRigs; // (a, b) rig indices
  for (const int b : grid) {
    std::vector<std::pair<double, int>> cand;
    for (const int64_t tA : matchSource.partnersBefore(rigSrcTime[b])) {
      const auto it = srcTimeToRig.find(tA);
      if (it == srcTimeToRig.end()) {
        continue;
      }
      const int a = it->second;
      const double dist = (pos(a) - pos(b)).norm();
      if ((rigs[b].timestampUs - rigs[a].timestampUs) * 1e-6 > s.candidateMinGapSec &&
          dist < s.candidateRadiusM) {
        cand.emplace_back(dist, a);
      }
    }
    std::sort(cand.begin(), cand.end());
    res.numCandidates += int(cand.size());
    std::vector<int> accepted;
    for (const auto& [dist, a] : cand) {
      if (int(accepted.size()) >= s.maxPartners) {
        break;
      }
      bool tooClose = false;
      for (const int other : accepted) {
        tooClose |= std::llabs(rigs[other].timestampUs - rigs[a].timestampUs) * 1e-6 <
            s.partnerSeparationSec;
      }
      if (tooClose) {
        continue;
      }
      LoopPair p;
      if (verify(a, b, p)) {
        accepted.push_back(a);
        pairRigs.emplace_back(a, b);
        res.pairs.push_back(std::move(p));
      }
    }
  }
  res.numVerified = int(res.pairs.size());
  say(fmt::format(
      "{} LC-KRs on the grid, {} pose-prior candidates, {} verified pairs",
      res.numGrid,
      res.numCandidates,
      res.numVerified));
  if (res.pairs.empty()) {
    res.ok = true;
    res.message = "no verified loop pair";
    return res;
  }

  // nodes: the grid and the partners
  std::set<int> nodeSet(grid.begin(), grid.end());
  for (const auto& [a, b] : pairRigs) {
    nodeSet.insert(a);
    nodeSet.insert(b);
  }
  res.nodeRig.assign(nodeSet.begin(), nodeSet.end());
  std::map<int, int> rigToNode;
  for (size_t n = 0; n < res.nodeRig.size(); n++) {
    const int i = res.nodeRig[n];
    rigToNode[i] = int(n);
    LcNode node;
    node.rigIndex = rigs[i].rigIndex;
    node.timestampUs = rigs[i].timestampUs;
    node.T_world_body = rigs[i].T_world_body;
    node.cameras = cameras(rigs[i].rigIndex);
    res.nodes.push_back(std::move(node));
  }
  for (size_t k = 0; k < res.pairs.size(); k++) {
    res.pairs[k].nodeA = rigToNode[pairRigs[k].first];
    res.pairs[k].nodeB = rigToNode[pairRigs[k].second];
  }
  const int nNodes = int(res.nodes.size());

  // odometry edges
  struct Odo {
    int i, j;
    OdometryNoiseModel::Sigmas sig;
  };
  std::vector<Odo> odo;
  for (int n = 0; n + 1 < nNodes; n++) {
    const int a = res.nodeRig[n], b = res.nodeRig[n + 1];
    double dist = 0.0;
    for (int r = a; r < b; r++) {
      dist += (pos(r + 1) - pos(r)).norm();
    }
    const double dt = (rigs[b].timestampUs - rigs[a].timestampUs) * 1e-6;
    // (without statistics: the reference quality of the model)
    const IntervalStats st = stats ? stats(rigs[a].timestampUs, rigs[b].timestampUs)
                                   : IntervalStats{s.odometry.refNumTracks, s.odometry.refTrackSec};
    odo.push_back({n, n + 1, s.odometry.sigmas(dt, dist, st)});
  }

  // the knot layout of the Knots variant
  KnotLayout layout;
  if (s.variant == PoseGraphVariant::Knots) {
    for (const auto& r : rigs) {
      layout.rigTimesUs.push_back(r.timestampUs);
      layout.rigPos.push_back(r.T_world_body.translation());
    }
    const double lengthSec = (rigs.back().timestampUs - rigs.front().timestampUs) * 1e-6;
    layout.knotTimeSec =
        s.knotTimesSec.size() >= 2 ? s.knotTimesSec : uniformKnotTimes(lengthSec, s.knotSec);
    layout.knotTimeSec.back() = std::max(layout.knotTimeSec.back(), lengthSec);
    layout.nodeRig = res.nodeRig;
    layout.rateSigma = s.knotRateSigma;
    layout.velSigma = s.knotVelSigma;
  }
  auto makeGraph = [&]() {
    return s.variant == PoseGraphVariant::Knots ? std::make_unique<PoseGraph>(res.nodes, layout)
                                                : std::make_unique<PoseGraph>(res.nodes);
  };
  auto states = [&](const PoseGraph& g) {
    std::vector<Vec4> out;
    for (int n = 0; n < g.numNodes(); n++) {
      out.push_back(g.nodeState(n));
    }
    return out;
  };

  // 2. stage 1
  auto g1 = makeGraph();
  for (const auto& e : odo) {
    g1->addOdometry(e.i, e.j, e.sig.yaw, e.sig.pos);
  }
  for (const auto& p : res.pairs) {
    const double kr = s.fitInflation * p.sigmaRotationRad, kd = s.fitInflation * p.sigmaDirectionRad;
    g1->addLoopEdge(
        p,
        std::sqrt(s.loopYawSigma * s.loopYawSigma + kr * kr),
        std::sqrt(s.loopDirSigma * s.loopDirSigma + kd * kd));
  }
  if (!g1->solve(s.gnc1, s.numThreads)) {
    res.message = "stage 1 failed";
    return res;
  }
  res.stage1 = states(*g1);
  int accepted1 = 0;
  for (int g = 0; g < g1->numGroups(); g++) {
    res.weights1.push_back(g1->groupWeight(g));
    accepted1 += g1->groupWeight(g) > 0.0;
  }
  say(fmt::format("stage 1: {} of {} pairs accepted", accepted1, res.pairs.size()));
  std::vector<Vec4> finalStates = res.stage1;
  const PoseGraph* finalGraph = g1.get();

  // 3. stage 2
  std::unique_ptr<PoseGraph> g2;
  if (s.stage2) {
    g2 = makeGraph();
    if (s.variant == PoseGraphVariant::Knots) {
      g2->setStartKnots(g1->knots());
    } else {
      std::vector<double> yaw;
      std::vector<Vec3> p;
      for (const auto& x : res.stage1) {
        yaw.push_back(x[0]);
        p.push_back(x.tail<3>());
      }
      g2->setStart(yaw, p);
    }
    for (const auto& e : odo) {
      g2->addOdometry(e.i, e.j, e.sig.yaw, e.sig.pos);
    }
    std::vector<int> pairOfGroup;
    res.weights2.assign(res.pairs.size(), -1.0);
    for (size_t k = 0; k < res.pairs.size(); k++) {
      if (res.weights1[k] <= 0.0) {
        continue;
      }
      LoopPair p = res.pairs[k];
      if (s.maxPointsPerPair > 0 && int(p.fA.size()) > s.maxPointsPerPair) {
        LoopPair sub = p;
        sub.pxA.clear();
        sub.pxB.clear();
        sub.fA.clear();
        sub.fB.clear();
        for (int m = 0; m < s.maxPointsPerPair; m++) {
          const size_t idx = size_t(m) * p.fA.size() / s.maxPointsPerPair;
          sub.pxA.push_back(p.pxA[idx]);
          sub.pxB.push_back(p.pxB[idx]);
          sub.fA.push_back(p.fA[idx]);
          sub.fB.push_back(p.fB[idx]);
        }
        p = std::move(sub);
      }
      if (g2->addLoopPoints(p, s.pixelSigma, s.huberPx) >= 0) {
        pairOfGroup.push_back(int(k));
      }
    }
    res.numPoints = g2->numPoints();
    if (!g2->solve(s.gnc2, s.numThreads)) {
      res.message = "stage 2 failed";
      return res;
    }
    res.stage2 = states(*g2);
    int accepted2 = 0;
    for (int g = 0; g < g2->numGroups(); g++) {
      res.weights2[pairOfGroup[g]] = g2->groupWeight(g);
      accepted2 += g2->groupWeight(g) > 0.0;
    }
    say(fmt::format(
        "stage 2: {} points, {} of {} pairs accepted", res.numPoints, accepted2, g2->numGroups()));
    finalStates = res.stage2;
    finalGraph = g2.get();
  }

  // 4. the correction
  if (s.variant == PoseGraphVariant::Knots) {
    DriftCorrection& c = res.correction;
    c.path.active = true;
    c.path.shift = true;
    for (const auto& r : rigs) {
      c.path.timeSec.push_back((r.timestampUs - rigs.front().timestampUs) * 1e-6);
      c.path.refPos.push_back(r.T_world_body.translation());
    }
    c.path.knotTimeSec = layout.knotTimeSec;
    const VecX& k = finalGraph->knots();
    for (int n = 0; n + 3 < int(k.size()); n += 4) {
      c.knots.push_back(k.segment<4>(n));
    }
  } else {
    std::vector<int64_t> times;
    std::vector<Vec3> p;
    for (const auto& r : rigs) {
      times.push_back(r.timestampUs);
      p.push_back(r.T_world_body.translation());
    }
    std::vector<double> nodeYaw;
    std::vector<Vec3> nodePos;
    for (const auto& x : finalStates) {
      nodeYaw.push_back(x[0]);
      nodePos.push_back(x.tail<3>());
    }
    res.correction = fitDriftCorrection(times, p, res.nodeRig, nodeYaw, nodePos);
  }
  res.ok = true;
  return res;
}

} // namespace visual_inertial_ba::loop
