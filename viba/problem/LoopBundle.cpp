/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/problem/LoopBundle.h>

#include <viba/common/Constants.h>
#include <viba/problem/VisualFactorRelative.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace visual_inertial_ba {

namespace {

Vec3 upCross(const Vec3& v) {
  return Vec3(-v.y(), v.x(), 0.0);
}

double medianOf(std::vector<double> v) {
  if (v.empty()) {
    return -1.0;
  }
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

// typed access to the variables of an observation (numeric Jacobians)
template <class T>
small_thing::Variable<T>* as(small_thing::VarBase* v) {
  return static_cast<small_thing::Variable<T>*>(v);
}

int tangentDim(small_thing::VarBase* v, LoopObservationStore::VarKind kind) {
  switch (kind) {
    case LoopObservationStore::kPoint:
      return 3;
    case LoopObservationStore::kPose:
      return 6;
    case LoopObservationStore::kKnot:
      return 4;
    default:
      return as<CameraModelParam>(v)->getTangentDim();
  }
}

void backupVar(small_thing::VarBase* v, LoopObservationStore::VarKind kind, VecX& d) {
  switch (kind) {
    case LoopObservationStore::kPoint:
      d = as<Vec3>(v)->value;
      break;
    case LoopObservationStore::kKnot:
      d = as<Vec4>(v)->value;
      break;
    case LoopObservationStore::kPose:
      d = Eigen::Map<const Vec7>(as<SE3>(v)->value.data());
      break;
    default: {
      auto* c = as<CameraModelParam>(v);
      d.resize(c->getDataDim());
      small_thing::VarSpec<CameraModelParam>::getData(c->value, d);
    }
  }
}

void restoreVar(small_thing::VarBase* v, LoopObservationStore::VarKind kind, const VecX& d) {
  switch (kind) {
    case LoopObservationStore::kPoint:
      as<Vec3>(v)->value = d;
      break;
    case LoopObservationStore::kKnot:
      as<Vec4>(v)->value = d;
      break;
    case LoopObservationStore::kPose:
      Eigen::Map<Vec7>(as<SE3>(v)->value.data()) = d;
      break;
    default:
      small_thing::VarSpec<CameraModelParam>::setData(as<CameraModelParam>(v)->value, d);
  }
}

void perturbVar(small_thing::VarBase* v, LoopObservationStore::VarKind kind, int t, double eps) {
  switch (kind) {
    case LoopObservationStore::kPoint:
      as<Vec3>(v)->value[t] += eps;
      break;
    case LoopObservationStore::kKnot:
      as<Vec4>(v)->value[t] += eps;
      break;
    case LoopObservationStore::kPose: {
      Vec6 s = Vec6::Zero();
      s[t] = eps;
      small_thing::VarSpec<SE3>::applyBoxPlus(as<SE3>(v)->value, s);
      break;
    }
    default: {
      auto* c = as<CameraModelParam>(v);
      VecX s = VecX::Zero(c->getTangentDim());
      s[t] = eps;
      small_thing::VarSpec<CameraModelParam>::applyBoxPlus(c->value, s);
    }
  }
}

} // namespace

// ---------------------------------------------------------------------------------------------
// LoopPathCache

double LoopPathCache::coef(int k, double t) const {
  const auto& T = path_->knotTimeSec;
  return T[k] < t ? std::min(t, T[k + 1]) - T[k] : 0.0;
}

void LoopPathCache::update(const YawDriftPath& path, const std::vector<KnotVariable*>& knots) {
  std::vector<double> key;
  key.reserve(4 * knots.size() + 4);
  for (const auto* k : knots) {
    key.insert(key.end(), k->value.data(), k->value.data() + 4);
  }
  // (the frozen copy changes when the knots are folded into the variables: its ends and size
  // are part of the key; the drift knots are folded with every rig moving)
  const size_t n = path.refPos.size();
  if (n > 0) {
    for (const size_t i : {size_t(0), n / 3, n / 2, n - 1}) {
      key.insert(key.end(), path.refPos[i].data(), path.refPos[i].data() + 3);
    }
  }
  key.push_back(double(n));
  key.push_back(path.shift ? 1.0 : 0.0);
  if (path_ == &path && key == key_) {
    return;
  }
  path_ = &path;
  key_ = key;
  const int K = int(knots.size());
  const int N = int(n);
  const auto& t = path.timeSec;
  const auto& ref = path.refPos;
  const auto& T = path.knotTimeSec;
  auto w = [&](int k) { return knots[k]->value[0]; };
  auto u = [&](int k) { return path.shift ? Vec3(knots[k]->value.tail<3>()) : Vec3(Vec3::Zero()); };
  psi_.resize(N);
  q_.resize(N);
  partial_.assign(N, {});
  first_.assign(K, -1);
  end_.assign(K, std::numeric_limits<int>::max());
  sumW_.assign(K, Vec3::Zero());
  sumU_.assign(K, Mat33::Zero());
  std::vector<Vec3> runW(K, Vec3::Zero());
  std::vector<Mat33> runU(K, Mat33::Zero());
  std::vector<int> active; // knots whose steps have started and not ended
  for (int i = 0; i < N; i++) {
    psi_[i] = path.psiAt(t[i], w);
  }
  q_[0] = ref[0];
  for (int j = 0; j + 1 < N; j++) {
    const int k0 = path.intervalOf(t[j]), k1 = path.intervalOf(t[j + 1]);
    Vec3 step = ref[j + 1] - ref[j];
    std::vector<std::pair<int, double>> parts; // (knot, a) of the partial knots
    std::vector<double> overlaps;
    for (int k = k0; k <= k1; k++) {
      const double L = T[k + 1] - T[k];
      const double cp = coef(k, t[j]), cn = coef(k, t[j + 1]);
      if (cp < L && cn > 0.0) {
        parts.emplace_back(k, 0.5 * (cp + cn));
        overlaps.push_back(cn - cp);
        step += (cn - cp) * u(k);
      }
    }
    const Mat33 R = SO3::rotZ(0.5 * (psi_[j] + psi_[j + 1])).matrix();
    const Vec3 r = R * step;
    q_[j + 1] = q_[j] + r;
    for (size_t p = 0; p < parts.size(); p++) {
      const int k = parts[p].first;
      if (first_[k] < 0) {
        first_[k] = j;
        active.push_back(k);
      }
      runW[k] += parts[p].second * upCross(r);
      runU[k] += overlaps[p] * R;
    }
    for (size_t a = 0; a < active.size();) {
      const int k = active[a];
      partial_[j + 1].push_back({k, runW[k], runU[k]});
      if (coef(k, t[j + 1]) >= T[k + 1] - T[k]) { // the interval is over
        end_[k] = j + 1;
        sumW_[k] = runW[k];
        sumU_[k] = runU[k];
        active.erase(active.begin() + a);
      } else {
        a++;
      }
    }
  }
}

Vec3 LoopPathCache::dqdw(int i, int k) const {
  if (first_[k] < 0 || i <= first_[k]) {
    return Vec3::Zero();
  }
  if (i <= end_[k]) {
    for (const auto& p : partial_[i]) {
      if (p.k == k) {
        return p.w;
      }
    }
    return Vec3::Zero();
  }
  const double L = path_->knotTimeSec[k + 1] - path_->knotTimeSec[k];
  return sumW_[k] + L * upCross(q_[i] - q_[end_[k]]);
}

Mat33 LoopPathCache::dqdu(int i, int k) const {
  if (first_[k] < 0 || i <= first_[k]) {
    return Mat33::Zero();
  }
  if (i <= end_[k]) {
    for (const auto& p : partial_[i]) {
      if (p.k == k) {
        return p.u;
      }
    }
    return Mat33::Zero();
  }
  return sumU_[k];
}

// ---------------------------------------------------------------------------------------------
// LoopObservationStore: set-up

LoopObservationStore::LoopObservationStore()
    : loss_(kReprojectionErrorHuberLossWidth, kReprojectionErrorHuberLossCutoff) {}

int LoopObservationStore::addPoint(const LoopPointParam& p) {
  points_.push_back(Point{p, {}, {}});
  return int(points_.size()) - 1;
}

void LoopObservationStore::addObservation(Observation&& o) {
  const Point& p = points_.at(o.point);
  o.vars.clear();
  o.kinds.clear();
  o.vars.push_back(p.param.var);
  o.kinds.push_back(kPoint);
  if (o.rig != p.param.host) {
    o.vars.push_back(p.param.host);
    o.kinds.push_back(kPose);
    o.vars.push_back(o.rig);
    o.kinds.push_back(kPose);
  }
  o.vars.push_back(o.T_cam);
  o.kinds.push_back(kPose);
  o.vars.push_back(o.cam);
  o.kinds.push_back(kCamera);
  if (o.rig != p.param.host) {
    for (int k = o.kLo; k <= o.kHi; k++) {
      o.vars.push_back(knots_.at(k));
      o.kinds.push_back(kKnot);
    }
  } else {
    o.kHi = o.kLo - 1;
  }
  pairLoss(o.pair); // (the pair exists)
  points_[o.point].obs.push_back(int(obs_.size()));
  obs_.push_back(std::move(o));
}

void LoopObservationStore::addVirtualFactor(
    int point,
    small_thing::VarBase* var,
    small_thing::FactorStoreBase* store,
    int64_t k) {
  auto& groups = points_.at(point).virtualGroups;
  for (auto& g : groups) {
    if (!g.empty() && g.front().var == var) {
      g.push_back({var, store, k});
      return;
    }
  }
  groups.push_back({{var, store, k}});
}

LoopPairLoss& LoopObservationStore::pairLoss(int pair) {
  while (int(pairWeights_.size()) <= pair) {
    pairWeights_.push_back(1.0);
    LoopPairLoss l;
    l.base = &loss_;
    l.w = &pairWeights_.back();
    pairLosses_.push_back(l);
  }
  return pairLosses_[pair];
}

void LoopObservationStore::setPairWeight(int pair, double w) {
  pairLoss(pair);
  pairWeights_[pair] = w;
}

double LoopObservationStore::pairWeight(int pair) const {
  return pair < int(pairWeights_.size()) ? pairWeights_[pair] : 1.0;
}

std::vector<double> LoopObservationStore::pairErrors() const {
  if (path_) {
    cache_.update(*path_, knots_);
  }
  std::vector<std::vector<double>> e(pairWeights_.size());
  for (const auto& o : obs_) {
    Vec2 err;
    if (evaluate(o, err, nullptr)) {
      e[o.pair].push_back(err.squaredNorm());
    }
  }
  std::vector<double> out;
  for (auto& v : e) {
    out.push_back(medianOf(v));
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// evaluation

bool LoopObservationStore::evaluate(
    const Observation& o,
    Vec2& err,
    Mat<2, Eigen::Dynamic>* J) const {
  const LoopPointParam& P = points_[o.point].param;
  const Vec3& pt = P.var->value;
  const CameraModelParam& cam = o.cam->value;
  const int nCam = o.cam->getTangentDim();
  int total = 0;
  for (size_t v = 0; v < o.vars.size(); v++) {
    total += tangentDim(o.vars[v], o.kinds[v]);
  }
  if (o.rig == P.host) {
    const VisualFactorHostRig f(o.px, o.sqrtH, P.inverseDepth, P.R0);
    if (J == nullptr) {
      const auto e = f(pt, o.T_cam->value, cam, small_thing::NullRef(), small_thing::NullRef(),
                       small_thing::NullRef());
      if (!e) {
        return false;
      }
      err = *e;
      return true;
    }
    Mat23 Jp;
    Mat26 Jc;
    Mat<2, Eigen::Dynamic> Jk = Mat<2, Eigen::Dynamic>::Zero(2, nCam);
    const auto e = f(pt, o.T_cam->value, cam, Jp, Jc, Jk);
    if (!e) {
      return false;
    }
    err = *e;
    J->setZero(2, total);
    J->leftCols<3>() = Jp;
    J->middleCols<6>(3) = Jc;
    J->middleCols(9, nCam) = Jk;
    return true;
  }

  // the observing rig B taken by H_A^-1 H_B (YawDrift.h), over every knot between A and B
  const SE3& T_B = o.rig->value;
  SE3 movedT_B = T_B;
  const bool drift = path_ != nullptr && path_->active && o.kHi >= o.kLo && o.rigA >= 0;
  double psiA = 0.0, d = 0.0;
  Vec3 m = Vec3::Zero(), lever = Vec3::Zero();
  SO3 movedR_B = T_B.so3();
  if (drift) {
    psiA = cache_.psi(o.rigA);
    d = cache_.psi(o.rigB) - psiA;
    m = SO3::rotZ(-psiA) * (cache_.q(o.rigB) - cache_.q(o.rigA));
    const Vec3 pos_B = -(T_B.so3().inverse() * T_B.translation());
    lever = SO3::rotZ(d) * (pos_B - path_->refPos[o.rigB]);
    const Vec3 movedPos = path_->refPos[o.rigA] + m + lever;
    movedR_B = T_B.so3() * SO3::rotZ(d).inverse();
    movedT_B = SE3(movedR_B, -(movedR_B * movedPos));
  }
  const VisualFactorHosted f(o.px, o.sqrtH, P.inverseDepth, P.R0);
  if (J == nullptr) {
    const auto e = f(pt, P.host->value, movedT_B, o.T_cam->value, cam, small_thing::NullRef(),
                     small_thing::NullRef(), small_thing::NullRef(), small_thing::NullRef(),
                     small_thing::NullRef());
    if (!e) {
      return false;
    }
    err = *e;
    return true;
  }
  Mat23 Jp;
  Mat26 JA, JB, Jc;
  Mat<2, Eigen::Dynamic> Jk = Mat<2, Eigen::Dynamic>::Zero(2, nCam);
  const auto e = f(pt, P.host->value, movedT_B, o.T_cam->value, cam, Jp, JA, JB, Jc, Jk);
  if (!e) {
    return false;
  }
  err = *e;
  J->setZero(2, total);
  J->leftCols<3>() = Jp;
  J->middleCols<6>(3) = JA;
  J->middleCols<6>(9) = JB;
  J->middleCols<6>(15) = Jc;
  J->middleCols(21, nCam) = Jk;
  if (drift) {
    const double tA = path_->timeSec[o.rigA], tB = path_->timeSec[o.rigB];
    const Mat33 RA = SO3::rotZ(-psiA).matrix();
    const Mat33 movedR = movedR_B.matrix();
    const Vec3 upInB = movedR * Vec3::UnitZ();
    const Vec3 dLever = upCross(lever);
    int col = 21 + nCam;
    for (int k = o.kLo; k <= o.kHi; k++, col += 4) {
      const double cA = cache_.coef(k, tA);
      const double cd = cache_.coef(k, tB) - cA;
      const Vec3 dm = RA * (cache_.dqdw(o.rigB, k) - cache_.dqdw(o.rigA, k)) - cA * upCross(m);
      const Vec3 dPos = dm + cd * dLever;
      J->col(col) = -(JB.leftCols<3>() * (movedR * dPos)) - cd * (JB.rightCols<3>() * upInB);
      if (path_->shift) {
        const Mat33 dmdu = RA * (cache_.dqdu(o.rigB, k) - cache_.dqdu(o.rigA, k));
        J->middleCols<3>(col + 1) = -(JB.leftCols<3>() * (movedR * dmdu));
      }
    }
  }
  return true;
}

double LoopObservationStore::observationCost(const Observation& o, const Vec2& err) const {
  return 0.5 * pairWeight(o.pair) * loss_.val(err.squaredNorm());
}

bool LoopObservationStore::unweightedError(int64_t i, Eigen::Ref<Eigen::VectorXd> err) const {
  if (path_) {
    cache_.update(*path_, knots_);
  }
  Vec2 e;
  if (!evaluate(obs_[i], e, nullptr)) {
    return true; // (as FactorStore: true when the factor fails)
  }
  err = e;
  return false;
}

double LoopObservationStore::unweightedSquaredError(int64_t i) const {
  if (path_) {
    cache_.update(*path_, knots_);
  }
  Vec2 e;
  return evaluate(obs_[i], e, nullptr) ? 0.5 * e.squaredNorm() : 0.0;
}

std::optional<double> LoopObservationStore::singleCost(int64_t i) const {
  if (path_) {
    cache_.update(*path_, knots_);
  }
  Vec2 e;
  if (!evaluate(obs_[i], e, nullptr)) {
    return {};
  }
  return observationCost(obs_[i], e);
}

std::optional<double> LoopObservationStore::varGradHess(
    int64_t i,
    int v,
    Eigen::Ref<Eigen::VectorXd> grad,
    Eigen::Ref<Eigen::MatrixXd> hess) const {
  if (path_) {
    cache_.update(*path_, knots_);
  }
  const Observation& o = obs_[i];
  Vec2 e;
  Mat<2, Eigen::Dynamic> J;
  if (!evaluate(o, e, &J)) {
    return {};
  }
  int col = 0;
  for (int u = 0; u < v; u++) {
    col += tangentDim(o.vars[u], o.kinds[u]);
  }
  const int dim = tangentDim(o.vars[v], o.kinds[v]);
  const auto [val, der] = loss_.jet2(e.squaredNorm());
  const double w = pairWeight(o.pair);
  const Mat<2, Eigen::Dynamic> Jv = J.middleCols(col, dim);
  grad = w * der * (Jv.transpose() * e);
  hess = w * der * (Jv.transpose() * Jv);
  return 0.5 * w * val;
}

std::tuple<double, double, double> LoopObservationStore::singleExpectedDelta(
    int64_t k,
    const double*,
    const BaSpaCho::PermutedCoalescedAccessor&) const {
  const auto c = singleCost(k);
  return {c ? *c : -1.0, 0.0, 0.0};
}

double LoopObservationStore::computeCost(
    bool makeComparableWithStored,
    small_thing::CostStats* stats,
    dispenso::ThreadPool*) const {
  if (path_) {
    cache_.update(*path_, knots_);
  }
  small_thing::CostStats st;
  double sum = 0.0;
  for (const auto& o : obs_) {
    st.numTotal++;
    Vec2 e;
    const bool ok = evaluate(o, e, nullptr);
    const bool prevInvalid = o.prevCost < 0.0;
    st.numInvalid += ok ? 0 : 1;
    st.numPrevInvalid += prevInvalid ? 1 : 0;
    if (makeComparableWithStored) {
      if (prevInvalid) {
        continue;
      }
      if (!ok) {
        sum += o.prevCost;
        continue;
      }
    }
    if (ok) {
      sum += observationCost(o, e);
    }
  }
  if (stats) {
    *stats = st;
  }
  return sum;
}

double LoopObservationStore::computeGradHess(
    double* gradData,
    const BaSpaCho::PermutedCoalescedAccessor& acc,
    double* /* hessData: never (matrix-free) */,
    bool updateCachedResults,
    bool dontRetryFailed,
    dispenso::ThreadPool*) const {
  if (path_) {
    cache_.update(*path_, knots_);
  }
  double sum = 0.0;
  Mat<2, Eigen::Dynamic> J;
  for (const auto& o : obs_) {
    if (dontRetryFailed && o.prevCost < 0.0) {
      continue;
    }
    Vec2 e;
    if (!evaluate(o, e, &J)) {
      if (updateCachedResults || dontRetryFailed) {
        o.prevCost = -1.0;
      }
      continue;
    }
    const auto [val, der] = loss_.jet2(e.squaredNorm());
    const double w = pairWeight(o.pair);
    const double cost = 0.5 * w * val;
    sum += cost;
    if (updateCachedResults) {
      o.prevCost = cost;
    }
    if (gradData) {
      int col = 0;
      for (size_t v = 0; v < o.vars.size(); v++) {
        const int dim = tangentDim(o.vars[v], o.kinds[v]);
        if (o.vars[v]->index >= 0) {
          Eigen::Map<VecX>(gradData + acc.paramStart(o.vars[v]->index), dim) +=
              (w * der) * (J.middleCols(col, dim).transpose() * e);
        }
        col += dim;
      }
    }
  }
  return sum;
}

void LoopObservationStore::registerVariables(
    std::vector<int64_t>& paramSizes,
    small_thing::TypedStore<small_thing::VariableStoreBase>& varStores) {
  auto reg = [&](auto* var) {
    using V = std::remove_pointer_t<decltype(var)>;
    if (var->index == small_thing::kUnsetIndex) {
      var->index = paramSizes.size();
      paramSizes.push_back(var->getTangentDim());
      varStores.get<small_thing::VariableStore<V>>().variables.push_back(var);
    }
  };
  for (auto& o : obs_) {
    for (size_t v = 0; v < o.vars.size(); v++) {
      switch (o.kinds[v]) {
        case kPoint:
          reg(as<Vec3>(o.vars[v]));
          break;
        case kPose:
          reg(as<SE3>(o.vars[v]));
          break;
        case kKnot:
          reg(as<Vec4>(o.vars[v]));
          break;
        default:
          reg(as<CameraModelParam>(o.vars[v]));
      }
    }
  }
}

bool LoopObservationStore::verifyJacobians(
    double epsilon,
    double relativeTolerance,
    double absoluteTolerance,
    bool stopAtFirstError,
    const small_thing::LogFunc& log) const {
  const size_t nCheck = std::min<size_t>(obs_.size(), 100);
  bool failed = false;
  double worst = 0.0;
  std::string worstWhere;
  for (size_t c = 0; c < nCheck && !(failed && stopAtFirstError); c++) {
    // (spread over all observations, so that the long ones are among them)
    const Observation& o = obs_[c * obs_.size() / nCheck];
    if (path_) {
      cache_.update(*path_, knots_);
    }
    Vec2 e0;
    Mat<2, Eigen::Dynamic> J;
    if (!evaluate(o, e0, &J)) {
      continue;
    }
    int col = 0;
    for (size_t v = 0; v < o.vars.size(); v++) {
      const int dim = tangentDim(o.vars[v], o.kinds[v]);
      VecX backup;
      backupVar(o.vars[v], o.kinds[v], backup);
      for (int t = 0; t < dim; t++) {
        perturbVar(o.vars[v], o.kinds[v], t, epsilon);
        if (path_) {
          cache_.update(*path_, knots_);
        }
        Vec2 e1;
        const bool ok = evaluate(o, e1, nullptr);
        restoreVar(o.vars[v], o.kinds[v], backup);
        if (!ok) {
          continue;
        }
        const Vec2 num = (e1 - e0) / epsilon;
        const double rel = std::max((num - J.col(col + t)).norm() - absoluteTolerance, 0.0) /
            (num.norm() + epsilon);
        if (rel > worst) {
          worst = rel;
          std::stringstream ss;
          ss << "observation " << (c * obs_.size() / nCheck) << ", variable " << v << " (kind "
             << o.kinds[v] << "), column " << t << ": analytic " << J.col(col + t).transpose()
             << ", numeric " << num.transpose();
          worstWhere = ss.str();
        }
        failed |= rel > relativeTolerance;
      }
      col += dim;
    }
  }
  if (path_) {
    cache_.update(*path_, knots_);
  }
  if (log) {
    std::stringstream ss;
    ss << "Verified factor class:\n" << name() << "\nFactors checked: " << nCheck << "/"
       << obs_.size() << ", Jacobian check " << (failed ? "FAILED" : "OK!")
       << " (relative tolerance: " << relativeTolerance << ", absolute tolerance: "
       << absoluteTolerance << "); largest relative error " << worst << " at " << worstWhere
       << "\n";
    log(ss.str());
  }
  return !failed;
}

// ---------------------------------------------------------------------------------------------
// MatrixFreeTerm

void LoopObservationStore::linearize(const BaSpaCho::PermutedCoalescedAccessor& acc, double damping) {
  if (path_) {
    cache_.update(*path_, knots_);
  }
  auto damp = [&](Mat33& H) {
    H.diagonal() *= 1.0 + damping;
    H.diagonal().array() += damping;
  };
  lin_.assign(points_.size(), PointLin());
  vlin_.assign(points_.size(), {});
  Mat<2, Eigen::Dynamic> J;
  for (size_t p = 0; p < points_.size(); p++) {
    const Point& P = points_[p];
    PointLin& L = lin_[p];
    L.slot = P.param.var->index >= 0 ? acc.paramStart(P.param.var->index) : -1;
    Mat33 H = Mat33::Zero();
    for (const int oi : P.obs) {
      const Observation& o = obs_[oi];
      Vec2 e;
      if (!evaluate(o, e, &J)) {
        continue; // (failed: no gradient either)
      }
      const double sw =
          std::sqrt(std::max(pairWeight(o.pair) * loss_.jet2(e.squaredNorm()).second, 0.0));
      Lin l;
      l.r = sw * e;
      l.Jp = sw * J.leftCols<3>();
      int col = 3, nx = 0;
      for (size_t v = 1; v < o.vars.size(); v++) {
        const int dim = tangentDim(o.vars[v], o.kinds[v]);
        if (o.vars[v]->index >= 0) {
          nx += dim;
        }
      }
      l.Jx.resize(2, nx);
      int xc = 0;
      for (size_t v = 1; v < o.vars.size(); v++) {
        const int dim = tangentDim(o.vars[v], o.kinds[v]);
        if (o.vars[v]->index >= 0) {
          l.cols.emplace_back(acc.paramStart(o.vars[v]->index), dim);
          l.Jx.middleCols(xc, dim) = sw * J.middleCols(col, dim);
          xc += dim;
        }
        col += dim;
      }
      H += l.Jp.transpose() * l.Jp;
      L.obs.push_back(std::move(l));
    }
    damp(H);
    L.Hinv.compute(H);
    // the virtual points of the groups: what the factorized matrix holds for them
    for (const auto& group : P.virtualGroups) {
      VirtualLin V;
      Mat33 Hv = Mat33::Zero();
      for (const auto& f : group) {
        std::vector<small_thing::VarBase*> vars;
        std::vector<int> dims;
        VecX we;
        MatX wJ;
        if (!f.store->linearizeFactor(f.k, vars, dims, we, wJ)) {
          continue;
        }
        Lin l;
        l.r = we;
        int nx = 0;
        for (size_t v = 0; v < vars.size(); v++) {
          nx += vars[v] == f.var ? 0 : dims[v];
        }
        l.Jx.resize(2, nx);
        int col = 0, xc = 0;
        for (size_t v = 0; v < vars.size(); v++) {
          if (vars[v] == f.var) {
            l.Jp = wJ.middleCols(col, 3);
          } else {
            l.cols.emplace_back(acc.paramStart(vars[v]->index), dims[v]);
            l.Jx.middleCols(xc, dims[v]) = wJ.middleCols(col, dims[v]);
            xc += dims[v];
          }
          col += dims[v];
        }
        Hv += l.Jp.transpose() * l.Jp;
        V.obs.push_back(std::move(l));
      }
      damp(Hv);
      V.Hinv.compute(Hv);
      vlin_[p].push_back(std::move(V));
    }
  }
}

void LoopObservationStore::addTrueHessianDense(Eigen::MatrixXd& H, double damping) const {
  for (const auto& L : lin_) {
    Mat33 Hp = Mat33::Zero();
    for (const auto& l : L.obs) {
      // columns: the point, then the reduced variables
      std::vector<std::pair<int64_t, int>> cols{{L.slot, 3}};
      cols.insert(cols.end(), l.cols.begin(), l.cols.end());
      Mat<2, Eigen::Dynamic> J(2, 3 + l.Jx.cols());
      J << l.Jp, l.Jx;
      int ci = 0;
      for (const auto& [oi, di] : cols) {
        int cj = 0;
        for (const auto& [oj, dj] : cols) {
          H.block(oi, oj, di, dj) += J.middleCols(ci, di).transpose() * J.middleCols(cj, dj);
          cj += dj;
        }
        ci += di;
      }
      Hp += l.Jp.transpose() * l.Jp;
    }
    H.block<3, 3>(L.slot, L.slot).diagonal() += damping * Hp.diagonal();
  }
}

void LoopObservationStore::prepareRhs(double* vec) {
  for (auto& L : lin_) {
    if (L.slot < 0) {
      continue;
    }
    L.g = Eigen::Map<Vec3>(vec + L.slot);
    Eigen::Map<Vec3>(vec + L.slot).setZero();
  }
}

void LoopObservationStore::correctReducedRhs(double* vec) {
  for (const auto& L : lin_) {
    const Vec3 t = L.Hinv.solve(L.g);
    for (const auto& l : L.obs) {
      const Vec2 u = l.Jp * t;
      int xc = 0;
      for (const auto& [off, dim] : l.cols) {
        Eigen::Map<VecX>(vec + off, dim) -= l.Jx.middleCols(xc, dim).transpose() * u;
        xc += dim;
      }
    }
  }
}

void LoopObservationStore::addPointProduct(
    const std::vector<Lin>& obs,
    const Eigen::LDLT<Mat33>& Hinv,
    const double* v,
    double* out,
    double sign) const {
  // S v = J_x^T J_x v - J_x^T J_p H^-1 J_p^T J_x v, over the observations of one point
  Vec3 a = Vec3::Zero();
  for (const auto& l : obs) {
    Vec2 u = Vec2::Zero();
    int xc = 0;
    for (const auto& [off, dim] : l.cols) {
      u += l.Jx.middleCols(xc, dim) * Eigen::Map<const VecX>(v + off, dim);
      xc += dim;
    }
    a += l.Jp.transpose() * u;
    xc = 0;
    for (const auto& [off, dim] : l.cols) {
      Eigen::Map<VecX>(out + off, dim) += sign * (l.Jx.middleCols(xc, dim).transpose() * u);
      xc += dim;
    }
  }
  const Vec3 t = Hinv.solve(a);
  for (const auto& l : obs) {
    const Vec2 z = l.Jp * t;
    int xc = 0;
    for (const auto& [off, dim] : l.cols) {
      Eigen::Map<VecX>(out + off, dim) -= sign * (l.Jx.middleCols(xc, dim).transpose() * z);
      xc += dim;
    }
  }
}

void LoopObservationStore::addMatvec(const double* v, double* out) {
  for (size_t p = 0; p < lin_.size(); p++) {
    addPointProduct(lin_[p].obs, lin_[p].Hinv, v, out, 1.0);
    for (const auto& V : vlin_[p]) {
      addPointProduct(V.obs, V.Hinv, v, out, -1.0);
    }
  }
}

void LoopObservationStore::backSubstitute(double* vec) {
  for (const auto& L : lin_) {
    if (L.slot < 0) {
      continue;
    }
    Vec3 b = L.g;
    for (const auto& l : L.obs) {
      Vec2 u = Vec2::Zero();
      int xc = 0;
      for (const auto& [off, dim] : l.cols) {
        u += l.Jx.middleCols(xc, dim) * Eigen::Map<const VecX>(vec + off, dim);
        xc += dim;
      }
      b -= l.Jp.transpose() * u;
    }
    Eigen::Map<Vec3>(vec + L.slot) = L.Hinv.solve(b);
  }
}

} // namespace visual_inertial_ba
