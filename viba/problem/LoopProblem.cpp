/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// SingleSessionProblem: loop-closing points (LoopBundle.h)

#include <viba/common/FindOrDie.h>
#include <viba/problem/SingleSessionProblem.h>
#include <viba/problem/VisualFactorRelative.h>

#include <logging/Checks.h>
#define DEFAULT_LOG_CHANNEL "ViBa::LoopProblem"
#include <logging/Log.h>

#include <algorithm>
#include <map>
#include <set>

namespace visual_inertial_ba {

LoopObservationStore* SingleSessionProblem::loopStore() {
  if (loopStore_ == nullptr) {
    loopStore_ = &opt_.factorStores.get<LoopObservationStore>();
  }
  std::vector<KnotVariable*> knots;
  for (auto& k : yawKnots_) {
    knots.push_back(&k);
  }
  loopStore_->setDrift(yawKnots_.empty() ? nullptr : &yawPath_, knots);
  return loopStore_;
}

Vec3 SingleSessionProblem::hostParamOf(const PointHost& host, const Vec3& worldPt) const {
  const Vec3 ptHost = findOrDie(inertialPoses_, host.rigIndex).T_bodyImu_world.value * worldPt;
  if (!host.inverseDepth) {
    return ptHost;
  }
  // R0 [u, v, 1] / rho = ptHost
  const Vec3 m = host.R0.transpose() * ptHost;
  const double rho = 1.0 / m.z();
  return Vec3(m.x() * rho, m.y() * rho, rho);
}

int64_t SingleSessionProblem::loopPoint_addNew(
    const Vec3& worldPt,
    int64_t hostRigIndex,
    bool inverseDepth) {
  PointHost host;
  host.rigIndex = hostRigIndex;
  host.inverseDepth = inverseDepth;
  if (inverseDepth) {
    const Vec3 ptHost =
        findOrDie(inertialPoses_, hostRigIndex).T_bodyImu_world.value * worldPt;
    const Vec3 d = ptHost.normalized();
    const Vec3 e1 = d.cross(std::abs(d[0]) < 0.9 ? Vec3::UnitX() : Vec3::UnitY()).normalized();
    host.R0.col(0) = e1;
    host.R0.col(1) = d.cross(e1);
    host.R0.col(2) = d;
  }
  loopPoints_.emplace_back(hostParamOf(host, worldPt));
  loopPointHosts_.push_back(host);
  loopObsRecords_.emplace_back();
  LoopPointParam p;
  p.var = &loopPoints_.back();
  p.host = &findOrDie(inertialPoses_, hostRigIndex).T_bodyImu_world;
  p.inverseDepth = inverseDepth;
  p.R0 = host.R0;
  const int index = loopStore()->addPoint(p);
  XR_CHECK_EQ(index, int64_t(loopPoints_.size()) - 1);
  return index;
}

Vec3 SingleSessionProblem::loopPoint_world(int64_t loopPoint) const {
  const Vec3& v = loopPoints_.at(loopPoint).value;
  const PointHost& host = loopPointHosts_.at(loopPoint);
  const Vec3 ptHost = host.inverseDepth ? Vec3(host.R0 * Vec3(v[0], v[1], 1.0) / v[2]) : v;
  return findOrDie(inertialPoses_, host.rigIndex).T_bodyImu_world.value.inverse() * ptHost;
}

void SingleSessionProblem::addLoopObservation(
    int64_t rigIndex,
    int64_t cameraIndex,
    int64_t loopPoint,
    const Vec2& projBaseRes,
    const Mat22& sqrtH_BaseRes,
    int visitGroup,
    int pairGroup) {
  auto* store = loopStore();
  auto& rig = findOrDie(inertialPoses_, rigIndex);
  const int64_t hostRig = loopPointHosts_.at(loopPoint).rigIndex;
  LoopObservationStore::Observation o;
  o.point = int(loopPoint);
  o.rig = &rig.T_bodyImu_world;
  o.T_cam = &T_Cam_BodyImu(rigIndex, int(cameraIndex)).var;
  o.cam = &cameraModel(rigIndex, int(cameraIndex)).var;
  o.px = projBaseRes;
  o.sqrtH = sqrtH_BaseRes;
  o.group = visitGroup;
  o.pair = pairGroup;
  if (!yawKnots_.empty() && rigIndex != hostRig) {
    o.rigA = findOrDie(yawRigOrder_, hostRig);
    o.rigB = findOrDie(yawRigOrder_, rigIndex);
    const double tA = yawPath_.timeSec[o.rigA], tB = yawPath_.timeSec[o.rigB];
    o.kLo = yawPath_.intervalOf(std::min(tA, tB));
    o.kHi = yawPath_.intervalOf(std::max(tA, tB));
  }
  store->addObservation(std::move(o));
  loopObsRecords_.at(loopPoint).push_back(
      {rigIndex, cameraIndex, rig.timestampUs, projBaseRes, sqrtH_BaseRes, visitGroup, pairGroup});
}

int64_t SingleSessionProblem::loopVirtual_build() {
  XR_CHECK(loopVirtualPoints_.empty(), "loopVirtual_build: called twice");
  auto* store = loopStore();
  for (size_t lp = 0; lp < loopPoints_.size(); lp++) {
    std::map<int, std::vector<const LoopObsRecord*>> groups;
    for (const auto& r : loopObsRecords_[lp]) {
      groups[r.visitGroup].push_back(&r);
    }
    for (auto& [_, recs] : groups) {
      std::set<int64_t> rigs;
      for (const auto* r : recs) {
        rigs.insert(r->rigIndex);
      }
      if (rigs.size() < 2) {
        continue; // cannot triangulate alone: contributes nothing to the preconditioner
      }
      std::sort(recs.begin(), recs.end(), [](const auto* a, const auto* b) {
        return a->timestampUs < b->timestampUs;
      });
      const Vec3 world = loopPoint_world(lp);
      LoopVirtual v;
      v.loopPoint = lp;
      v.host = loopPointHosts_[lp];
      v.host.rigIndex = recs[recs.size() / 2]->rigIndex;
      if (v.host.inverseDepth) {
        const Vec3 d = (findOrDie(inertialPoses_, v.host.rigIndex).T_bodyImu_world.value * world)
                           .normalized();
        const Vec3 e1 = d.cross(std::abs(d[0]) < 0.9 ? Vec3::UnitX() : Vec3::UnitY()).normalized();
        v.host.R0.col(0) = e1;
        v.host.R0.col(1) = d.cross(e1);
        v.host.R0.col(2) = d;
      }
      loopVirtualPoints_.emplace_back(hostParamOf(v.host, world));
      loopVirtualInfo_.push_back(v);
      auto& vpt = loopVirtualPoints_.back();
      auto& hostVar = findOrDie(inertialPoses_, v.host.rigIndex);
      for (const auto* r : recs) {
        auto& rigVar = findOrDie(inertialPoses_, r->rigIndex);
        auto& extr = T_Cam_BodyImu(r->rigIndex, int(r->cameraIndex)).var;
        auto& cam = cameraModel(r->rigIndex, int(r->cameraIndex)).var;
        const LoopPairLoss& loss = store->pairLoss(r->pairGroup);
        if (r->rigIndex == v.host.rigIndex) {
          auto [st, k] = opt_.addFactor(
              small_thing::PreconditionerOnly<VisualFactorHostRig>(
                  VisualFactorHostRig(r->px, r->sqrtH, v.host.inverseDepth, v.host.R0)),
              loss,
              vpt,
              extr,
              cam);
          store->addVirtualFactor(int(lp), &vpt, st, k);
          continue;
        }
        const VisualFactorHosted f(r->px, r->sqrtH, v.host.inverseDepth, v.host.R0);
        const int a = yawKnots_.empty() ? 0 : findOrDie(yawRigOrder_, v.host.rigIndex);
        const int b = yawKnots_.empty() ? 0 : findOrDie(yawRigOrder_, r->rigIndex);
        const int knot = yawKnots_.empty() ? -1 : yawPath_.knotPairFor(a, b);
        if (knot >= 0) {
          using Wrapped = YawDrifted<
              VisualFactorHosted, 2, 1, -1, 2, -1, //
              Vec3, SE3, SE3, SE3, CameraModelParam>;
          auto [st, k] = opt_.addFactor(
              small_thing::PreconditionerOnly<Wrapped>(Wrapped(f, &yawPath_, a, b, knot)),
              loss,
              vpt,
              hostVar.T_bodyImu_world,
              rigVar.T_bodyImu_world,
              extr,
              cam,
              yawKnots_[knot],
              yawKnots_[knot + 1]);
          store->addVirtualFactor(int(lp), &vpt, st, k);
        } else {
          auto [st, k] = opt_.addFactor(
              small_thing::PreconditionerOnly<VisualFactorHosted>(VisualFactorHosted(f)),
              loss,
              vpt,
              hostVar.T_bodyImu_world,
              rigVar.T_bodyImu_world,
              extr,
              cam);
          store->addVirtualFactor(int(lp), &vpt, st, k);
        }
      }
    }
  }
  if (verbosity_ != Muted) {
    XR_LOGI(
        "Loop points: {}, observations {}, virtual points {}",
        loopPoints_.size(),
        store->observations().size(),
        loopVirtualPoints_.size());
  }
  return loopVirtualPoints_.size();
}

void SingleSessionProblem::loopVirtual_sync() {
  for (size_t i = 0; i < loopVirtualPoints_.size(); i++) {
    const auto& v = loopVirtualInfo_[i];
    loopVirtualPoints_[i].value = hostParamOf(v.host, loopPoint_world(v.loopPoint));
  }
}

} // namespace visual_inertial_ba
