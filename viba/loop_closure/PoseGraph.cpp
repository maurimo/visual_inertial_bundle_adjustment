/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/loop_closure/PoseGraph.h>

#include <viba/problem/VisualFactorRelative.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace visual_inertial_ba::loop {

namespace {

Mat33 rotZ(double psi) {
  const double c = std::cos(psi), s = std::sin(psi);
  Mat33 R;
  R << c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0;
  return R;
}

// z x v
Vec3 upCross(const Vec3& v) {
  return Vec3(-v.y(), v.x(), 0.0);
}
Mat33 upCrossMat() {
  Mat33 m;
  m << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0;
  return m;
}

void tangentBasis(const Vec3& v, Vec3& b1, Vec3& b2) {
  const Vec3 a = std::abs(v.x()) < 0.9 ? Vec3::UnitX() : Vec3::UnitY();
  b1 = v.cross(a).normalized();
  b2 = v.cross(b1).normalized();
}

// rotation whose third column is the unit vector d
Mat33 bearingFrame(const Vec3& d) {
  Vec3 e1, e2;
  tangentBasis(d, e1, e2);
  Mat33 R0;
  R0.col(0) = e1;
  R0.col(1) = d.cross(e1);
  R0.col(2) = d;
  return R0;
}

double median(std::vector<double> v) {
  if (v.empty()) {
    return 0.0;
  }
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

} // namespace

double yawOf(const Mat33& R) {
  return std::atan2(R(1, 0) - R(0, 1), R(0, 0) + R(1, 1));
}

// ---------------------------------------------------------------------------------------------

Vec4 OdometryFactor::operator()(
    const Vec4& xi,
    const Vec4& xj,
    Ref<Mat<4, 4>> Ji,
    Ref<Mat<4, 4>> Jj) const {
  const double c = std::cos(xi[0]), s = std::sin(xi[0]);
  const Vec3 dp = xj.tail<3>() - xi.tail<3>();
  Mat33 Rt; // Rz(-psi_i)
  Rt << c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0;
  const Vec3 r = Rt * dp;
  Vec4 e;
  e[0] = wYaw_ * (xj[0] - xi[0]);
  e.tail<3>() = wPos_.cwiseProduct(r - dp0_);
  if (!small_thing::isNull(Ji)) {
    Ji.setZero();
    Ji(0, 0) = -wYaw_;
    Ji.block<3, 1>(1, 0) = wPos_.cwiseProduct(Vec3(r.y(), -r.x(), 0.0));
    Ji.block<3, 3>(1, 1) = -(wPos_.asDiagonal() * Rt);
  }
  if (!small_thing::isNull(Jj)) {
    Jj.setZero();
    Jj(0, 0) = wYaw_;
    Jj.block<3, 3>(1, 1) = wPos_.asDiagonal() * Rt;
  }
  return e;
}

// ---------------------------------------------------------------------------------------------

LoopEdgeFactor::LoopEdgeFactor(
    const Mat33& R0i,
    const Mat33& R0j,
    const SE3& T_ca_body,
    const SE3& T_cb_body,
    const RelativePose& meas,
    bool directionValid,
    double sigmaYaw,
    double sigmaDir)
    : R0i_(R0i),
      R0j_(R0j),
      R_ca_body_(T_ca_body.so3().matrix()),
      la_(T_ca_body.inverse().translation()),
      lb_(T_cb_body.inverse().translation()),
      directionValid_(directionValid),
      wYaw_(1.0 / sigmaYaw),
      wDir_(1.0 / sigmaDir) {
  const Mat33 R_body_ca = T_ca_body.so3().inverse().matrix();
  const Mat33 R_cb_body = T_cb_body.so3().matrix();
  // measured R_world_cb = R_world_ca R^T; predicted R_world_cb = R_world_j R_body_cb; their
  // difference R_meas R_pred^T = Rz(psi_i) [R0i R_body_ca R^T R_cb_body R0j^T] Rz(-psi_j)
  yawMeas_ = yawOf(R0i * R_body_ca * meas.R.transpose() * R_cb_body * R0j.transpose());
  tangentBasis(meas.directionInCamera1().normalized(), b1_, b2_);
}

Vec3 LoopEdgeFactor::operator()(
    const Vec4& xi,
    const Vec4& xj,
    Ref<Mat<3, 4>> Ji,
    Ref<Mat<3, 4>> Jj) const {
  Vec3 e = Vec3::Zero();
  e[0] = wYaw_ * (xi[0] - xj[0] + yawMeas_);
  const bool needJ = !small_thing::isNull(Ji) || !small_thing::isNull(Jj);
  Mat<3, 4> Ai = Mat<3, 4>::Zero(), Aj = Mat<3, 4>::Zero();
  Ai(0, 0) = wYaw_;
  Aj(0, 0) = -wYaw_;
  if (directionValid_) {
    const Mat33 Ri = rotZ(xi[0]) * R0i_;
    const Mat33 Rj = rotZ(xj[0]) * R0j_;
    const Vec3 Rila = Ri * la_, Rjlb = Rj * lb_;
    const Vec3 w = xj.tail<3>() + Rjlb - xi.tail<3>() - Rila; // c_j - c_i
    const Mat33 M = R_ca_body_ * Ri.transpose();
    const Vec3 v = M * w;
    const double nv = v.norm();
    if (nv > 1e-9) {
      const Vec3 n = v / nv;
      e[1] = wDir_ * b1_.dot(n);
      e[2] = wDir_ * b2_.dot(n);
      if (needJ) {
        Mat<2, 3> P;
        P.row(0) = b1_.transpose();
        P.row(1) = b2_.transpose();
        const Mat<2, 3> D = wDir_ * P * (Mat33::Identity() - n * n.transpose()) / nv;
        const Mat<2, 3> DM = D * M;
        Ai.block<2, 1>(1, 0) = DM * (-upCross(w + Rila));
        Ai.block<2, 3>(1, 1) = -DM;
        Aj.block<2, 1>(1, 0) = DM * upCross(Rjlb);
        Aj.block<2, 3>(1, 1) = DM;
      }
    }
  }
  if (!small_thing::isNull(Ji)) {
    Ji = Ai;
  }
  if (!small_thing::isNull(Jj)) {
    Jj = Aj;
  }
  return e;
}

// ---------------------------------------------------------------------------------------------

std::optional<Vec2> LoopPointHostObsFactor::operator()(const Vec3& pt, Ref<Mat23> Jpt) const {
  const Vec3 m = R0_ * Vec3(pt[0], pt[1], 1.0);
  Vec2 err;
  Mat23 A;
  Mat<2, Eigen::Dynamic> dCam;
  if (!relpoint::projectScaled(*cam_, m, px_, sqrtH_, err, A, dCam)) {
    return {};
  }
  if (!small_thing::isNull(Jpt)) {
    Jpt.col(0) = A * R0_.col(0);
    Jpt.col(1) = A * R0_.col(1);
    Jpt.col(2).setZero();
  }
  return err;
}

LoopPointObsFactor::LoopPointObsFactor(
    const CameraModelParam* camB,
    const Mat33& R0,
    const Mat33& R0i,
    const Mat33& R0j,
    const SE3& T_ca_body,
    const SE3& T_cb_body,
    const Vec2& px,
    double sqrtInfo)
    : camB_(camB),
      R0_(R0),
      R0i_(R0i),
      R0j_(R0j),
      R_body_ca_(T_ca_body.so3().inverse().matrix()),
      R_cb_body_(T_cb_body.so3().matrix()),
      la_(T_ca_body.inverse().translation()),
      lb_(T_cb_body.inverse().translation()),
      px_(px),
      sqrtH_(Mat22::Identity() * sqrtInfo) {}

std::optional<Vec2> LoopPointObsFactor::operator()(
    const Vec3& pt,
    const Vec4& xi,
    const Vec4& xj,
    Ref<Mat23> Jpt,
    Ref<Mat<2, 4>> Ji,
    Ref<Mat<2, 4>> Jj) const {
  const double rho = pt[2];
  const Vec3 m = R0_ * Vec3(pt[0], pt[1], 1.0);
  const Mat33 Ri = rotZ(xi[0]) * R0i_;
  const Mat33 Rj = rotZ(xj[0]) * R0j_;
  const Vec3 ci = xi.tail<3>() + Ri * la_;
  const Vec3 Rjlb = Rj * lb_;
  const Vec3 cj = xj.tail<3>() + Rjlb;
  const Vec3 dir = Ri * (R_body_ca_ * m); // world direction of the point from camera a
  const Vec3 g = dir + rho * (ci - cj); // rho * (point - c_j)
  const Mat33 Mj = R_cb_body_ * Rj.transpose();
  const Vec3 q = Mj * g;
  Vec2 err;
  Mat23 A;
  Mat<2, Eigen::Dynamic> dCam;
  if (!relpoint::projectScaled(*camB_, q, px_, sqrtH_, err, A, dCam)) {
    return {};
  }
  const Mat23 AM = A * Mj;
  if (!small_thing::isNull(Jpt)) {
    const Mat33 RR = Ri * R_body_ca_;
    Jpt.col(0) = AM * (RR * R0_.col(0));
    Jpt.col(1) = AM * (RR * R0_.col(1));
    Jpt.col(2) = AM * (ci - cj);
  }
  if (!small_thing::isNull(Ji)) {
    Ji.col(0) = AM * upCross(Ri * (R_body_ca_ * m + rho * la_));
    Ji.rightCols<3>() = rho * AM;
  }
  if (!small_thing::isNull(Jj)) {
    Jj.col(0) = AM * (-upCross(g) - rho * upCross(Rjlb));
    Jj.rightCols<3>() = -rho * AM;
  }
  return err;
}

// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------
// Knots variant: node states as functions of the knots

KnotNodeStates::KnotNodeStates(const KnotLayout& layout) : layout_(layout) {}

void KnotNodeStates::update(const VecX& knots) const {
  if (cachedKnots_.size() == knots.size() && !states_.empty() && cachedKnots_ == knots) {
    return;
  }
  const KnotLayout& L = layout_;
  const int K = numKnots();
  const int N = int(L.rigTimesUs.size());
  const auto& T = L.knotTimeSec;
  // c_k(t): the part of [T_k, T_{k+1}] before t (Psi(t) = sum_k w_k c_k(t), YawDriftPath::psiAt)
  auto coefs = [&](double t, std::vector<double>& c) {
    c.resize(K);
    for (int k = 0; k < K; k++) {
      c[k] = T[k] < t ? std::min(t, T[k + 1]) - T[k] : 0.0;
    }
  };
  states_.assign(L.nodeRig.size(), Vec4::Zero());
  jacobians_.assign(L.nodeRig.size(), Mat<4, Eigen::Dynamic>::Zero(4, 4 * K));
  std::vector<double> cPrev, cNext;
  coefs(0.0, cPrev);
  double psiPrev = 0.0;
  Vec3 q = L.rigPos[0];
  Mat<3, Eigen::Dynamic> Dq = Mat<3, Eigen::Dynamic>::Zero(3, 4 * K);
  size_t node = 0;
  for (int i = 0; i < N; i++) {
    const double ti = (L.rigTimesUs[i] - L.rigTimesUs[0]) * 1e-6;
    if (i > 0) {
      coefs(ti, cNext);
      double psiNext = 0.0;
      for (int k = 0; k < K; k++) {
        psiNext += knots[4 * k] * cNext[k];
      }
      Vec3 step = L.rigPos[i] - L.rigPos[i - 1];
      for (int k = 0; k < K; k++) {
        const double overlap = cNext[k] - cPrev[k];
        if (overlap != 0.0) {
          step += overlap * knots.segment<3>(4 * k + 1);
        }
      }
      const Mat33 R = rotZ(0.5 * (psiPrev + psiNext));
      const Vec3 r = R * step;
      const Vec3 zr = upCross(r);
      for (int k = 0; k < K; k++) {
        const double a = 0.5 * (cPrev[k] + cNext[k]);
        if (a != 0.0) {
          Dq.col(4 * k) += a * zr;
        }
        const double overlap = cNext[k] - cPrev[k];
        if (overlap != 0.0) {
          Dq.block(0, 4 * k + 1, 3, 3) += overlap * R;
        }
      }
      q += r;
      psiPrev = psiNext;
      cPrev.swap(cNext);
    }
    while (node < L.nodeRig.size() && L.nodeRig[node] == i) {
      states_[node] = Vec4(psiPrev, q.x(), q.y(), q.z());
      for (int k = 0; k < K; k++) {
        jacobians_[node](0, 4 * k) = cPrev[k];
      }
      jacobians_[node].bottomRows<3>() = Dq;
      node++;
    }
  }
  cachedKnots_ = knots;
}

const Vec4& KnotNodeStates::state(const VecX& knots, int node) const {
  update(knots);
  return states_[node];
}

const Mat<4, Eigen::Dynamic>& KnotNodeStates::jacobian(const VecX& knots, int node) const {
  update(knots);
  return jacobians_[node];
}

std::optional<Vec2> KnotPointObsFactor::operator()(
    const Vec3& pt,
    const VecX& knots,
    Ref<Mat23> Jpt,
    Ref<Mat<2, Eigen::Dynamic>> J) const {
  const Vec4& xi = states_->state(knots, i_);
  const Vec4& xj = states_->state(knots, j_);
  if (small_thing::isNull(J)) {
    return base_(pt, xi, xj, Jpt, small_thing::NullRef(), small_thing::NullRef());
  }
  Mat<2, 4> Ji, Jj;
  auto e = base_(pt, xi, xj, Jpt, Ji, Jj);
  if (e.has_value()) {
    J = Ji * states_->jacobian(knots, i_) + Jj * states_->jacobian(knots, j_);
  }
  return e;
}

// ---------------------------------------------------------------------------------------------

PoseGraph::PoseGraph(const std::vector<LcNode>& nodes) : nodes_(nodes) {
  vars_.reserve(nodes_.size());
  for (const auto& n : nodes_) {
    vars_.emplace_back(Vec4(0.0, n.T_world_body.translation().x(),
                            n.T_world_body.translation().y(),
                            n.T_world_body.translation().z()));
  }
  if (!vars_.empty()) {
    vars_[0].setConstant(); // gauge
  }
}

PoseGraph::PoseGraph(const std::vector<LcNode>& nodes, const KnotLayout& layout)
    : nodes_(nodes), knotStates_(std::make_shared<KnotNodeStates>(layout)) {
  if (layout.nodeRig.size() != nodes_.size() || layout.knotTimeSec.size() < 2 ||
      layout.rigTimesUs.size() != layout.rigPos.size()) {
    throw std::invalid_argument("PoseGraph: inconsistent knot layout");
  }
  const int K = knotStates_->numKnots();
  knotVar_.value = VecX::Zero(4 * K);
  for (int k = 0; k < K; k++) {
    opt_.addFactor(KnotPriorFactor(k, K, layout.rateSigma, layout.velSigma), knotVar_);
  }
}

void PoseGraph::setStart(const std::vector<double>& yaw, const std::vector<Vec3>& pos) {
  if (knotStates_) {
    return;
  }
  for (size_t i = 1; i < vars_.size(); i++) { // (node 0 is held)
    vars_[i].value[0] = yaw[i];
    vars_[i].value.tail<3>() = pos[i];
  }
}

void PoseGraph::setStartKnots(const VecX& knots) {
  if (knotStates_ && knots.size() == knotVar_.value.size()) {
    knotVar_.value = knots;
  }
}

Vec4 PoseGraph::nodeState(int i) const {
  return knotStates_ ? knotStates_->state(knotVar_.value, i) : vars_[i].value;
}

SE3 PoseGraph::T_world_body(int i) const {
  const Mat33 R = rotZ(yaw(i)) * nodes_[i].T_world_body.so3().matrix();
  return SE3(SO3(Eigen::Quaterniond(R).normalized()), position(i));
}

// (factors are passed to addFactor as temporaries: a named one would be stored by reference)
void PoseGraph::addOdometry(int i, int j, double sigmaYaw, const Vec3& sigmaPos) {
  const Vec3 dp0 = nodes_[j].T_world_body.translation() - nodes_[i].T_world_body.translation();
  const OdometryFactor f(dp0, sigmaYaw, sigmaPos);
  if (knotStates_) {
    opt_.addFactor(KnotPairFactor<OdometryFactor, 4>(f, knotStates_, i, j), knotVar_);
  } else {
    opt_.addFactor(OdometryFactor(f), vars_[i], vars_[j]);
  }
}

int PoseGraph::addLoopEdge(const LoopPair& pair, double sigmaYaw, double sigmaDir) {
  const int i = pair.nodeA, j = pair.nodeB;
  Group g;
  g.loss = std::make_unique<GncLoss>(0.0);
  const LoopEdgeFactor f(
      nodes_[i].T_world_body.so3().matrix(),
      nodes_[j].T_world_body.so3().matrix(),
      nodes_[i].cameras.at(pair.camA).T_cam_body,
      nodes_[j].cameras.at(pair.camB).T_cam_body,
      pair.pose,
      pair.directionValid,
      sigmaYaw,
      sigmaDir);
  if (knotStates_) {
    const auto [store, k] =
        opt_.addFactor(KnotPairFactor<LoopEdgeFactor, 3>(f, knotStates_, i, j), *g.loss, knotVar_);
    g.factors.emplace_back(store, k);
  } else {
    const auto [store, k] = opt_.addFactor(LoopEdgeFactor(f), *g.loss, vars_[i], vars_[j]);
    g.factors.emplace_back(store, k);
  }
  groups_.push_back(std::move(g));
  return int(groups_.size()) - 1;
}

int PoseGraph::addLoopPoints(const LoopPair& pair, double pixelSigma, double huberPx) {
  if (registered_) {
    return -1; // (points have to exist before the first solve: they are the elimination range)
  }
  const int i = pair.nodeA, j = pair.nodeB;
  const CameraAtRig& ca = nodes_[i].cameras.at(pair.camA);
  const CameraAtRig& cb = nodes_[j].cameras.at(pair.camB);
  const Mat33 R0i = nodes_[i].T_world_body.so3().matrix();
  const Mat33 R0j = nodes_[j].T_world_body.so3().matrix();
  // current node values, for the triangulation
  const Mat33 Ri = rotZ(yaw(i)) * R0i, Rj = rotZ(yaw(j)) * R0j;
  const Vec3 centreA = position(i) + Ri * ca.T_cam_body.inverse().translation();
  const Vec3 centreB = position(j) + Rj * cb.T_cam_body.inverse().translation();
  const Mat33 Rwa = Ri * ca.T_cam_body.so3().inverse().matrix();
  const Mat33 Rwb = Rj * cb.T_cam_body.so3().inverse().matrix();
  Group g;
  g.loss = std::make_unique<GncLoss>(huberPx / pixelSigma);
  for (size_t k = 0; k < pair.fA.size(); k++) {
    const Vec3 fa = pair.fA[k].normalized(), fb = pair.fB[k].normalized();
    // midpoint triangulation: da ra - db rb = centreB - centreA
    const Vec3 ra = Rwa * fa, rb = Rwb * fb, d = centreB - centreA;
    const double aa = 1.0, bb = 1.0, ab = -ra.dot(rb);
    const double det = aa * bb - ab * ab;
    double rho = 1e-3; // far (1 km) when there is no usable triangulation
    if (det > 1e-8) {
      const double da = (d.dot(ra) * bb - ab * (-d.dot(rb))) / det;
      const double db = (aa * (-d.dot(rb)) - ab * d.dot(ra)) / det;
      if (da > 0.3 && db > 0.3) {
        rho = 1.0 / da;
      }
    }
    const Mat33 R0 = bearingFrame(fa);
    points_.emplace_back(Vec3(0.0, 0.0, rho));
    auto& pt = points_.back();
    auto [sa, ka] = opt_.addFactor(
        LoopPointHostObsFactor(ca.model, R0, pair.pxA[k], 1.0 / pixelSigma), *g.loss, pt);
    const LoopPointObsFactor obsB(
        cb.model, R0, R0i, R0j, ca.T_cam_body, cb.T_cam_body, pair.pxB[k], 1.0 / pixelSigma);
    g.factors.emplace_back(sa, ka);
    if (knotStates_) {
      const auto [sb, kb] =
          opt_.addFactor(KnotPointObsFactor(obsB, knotStates_, i, j), *g.loss, pt, knotVar_);
      g.factors.emplace_back(sb, kb);
    } else {
      const auto [sb, kb] = opt_.addFactor(LoopPointObsFactor(obsB), *g.loss, pt, vars_[i], vars_[j]);
      g.factors.emplace_back(sb, kb);
    }
  }
  if (g.factors.empty()) {
    return -1;
  }
  groups_.push_back(std::move(g));
  return int(groups_.size()) - 1;
}

double PoseGraph::groupError(int g) const {
  std::vector<double> e;
  for (const auto& [store, k] : groups_[g].factors) {
    e.push_back(2.0 * store->unweightedSquaredError(k));
  }
  return median(e);
}

void PoseGraph::registerOnce() {
  if (registered_) {
    return;
  }
  for (auto& pt : points_) {
    opt_.registerVariable(pt);
  }
  opt_.registeredVariablesToEliminationRange();
  registered_ = true;
}

bool PoseGraph::solve(const GncSettings& gnc, int numThreads) {
  registerOnce();
  small_thing::Optimizer::Settings s;
  s.numThreads = numThreads;
  s.log = nullptr;
  s.damping = 1e-4;
  s.dampingMin = 1e-12;
  // (as in the real-data runs, made when the environment's VIBA_DAMPING_GOOD=0.1 set this for every
  // optimizer call)
  s.dampingAdjustOnGoodStep = 0.1;
  auto run = [&](int iterations) {
    s.maxNumIterations = iterations;
    const auto summary = opt_.optimize(s);
    return std::isfinite(summary.finalCost);
  };
  for (auto& g : groups_) {
    g.loss->w = 1.0;
  }
  if (!run(gnc.innerIterations)) {
    return false;
  }
  if (gnc.enabled && !groups_.empty()) {
    const double c2 = gnc.thresholdSq;
    std::vector<double> r2(groups_.size());
    double maxR2 = 0.0;
    for (size_t g = 0; g < groups_.size(); g++) {
      r2[g] = groupError(int(g));
      maxR2 = std::max(maxR2, r2[g]);
    }
    double mu = 2.0 * maxR2 / c2;
    while (mu > 1.0) {
      for (size_t g = 0; g < groups_.size(); g++) {
        const double a = mu * c2 / (r2[g] + mu * c2);
        groups_[g].loss->w = a * a;
      }
      if (!run(gnc.innerIterations)) {
        return false;
      }
      for (size_t g = 0; g < groups_.size(); g++) {
        r2[g] = groupError(int(g));
      }
      mu /= gnc.muFactor;
    }
    // the exact form: accepted pairs with weight 1, the others switched off
    for (size_t g = 0; g < groups_.size(); g++) {
      groups_[g].loss->w = r2[g] < c2 ? 1.0 : 0.0;
    }
  }
  return run(gnc.finalIterations);
}

} // namespace visual_inertial_ba::loop
