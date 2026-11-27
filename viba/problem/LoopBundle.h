/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <camera_model/CameraModelParam.h>
#include <small_thing/Factor.h>
#include <small_thing/MatrixFree.h>
#include <small_thing/SoftLoss.h>
#include <viba/problem/Types.h>
#include <viba/problem/YawDrift.h>

#include <deque>
#include <optional>
#include <vector>

// Loop-closed bundle adjustment (LC_METHODS.md 7.1 with the addendum; LC_IMPL.md in the main
// repository): one point variable per loop landmark, with all its observations, minutes apart.
//
// - The observations are in one store of role kRoleMatrixFree: cost and gradient as for any
//   factor, but their curvature is NOT in the factorized matrix (it would join rigs minutes apart
//   and fill the band). The store is the MatrixFreeTerm of the PCG solver: it eliminates the loop
//   points itself and adds their reduced curvature to the product of the PCG.
// - Per loop point and visit group that can triangulate it alone (observations from 2 rigs at
//   least), a "virtual point": a variable of the elimination range with preconditioner-only copies
//   of that group's observations, hosted in a rig of the group, so that the factorized matrix gets
//   the group's banded curvature S_g. The PCG product subtracts exactly what the virtual points put
//   into the factor (their linearization, damped as the solver damps them).
// - The drift knots: an observation joins its host and its rig through every knot interval
//   between them; the knot wrap of YawDrift.h is generalized to any number of intervals
//   (LoopPathCache), so the knots stay a neutral over-parametrization (VIBA_YAW_GAUGE_TEST) for
//   these factors too.
// - Robustness: a weight per loop pair (GNC), set from outside between iterations, on top of the
//   store's own robust loss (whose size can also be changed between iterations).
namespace visual_inertial_ba {

using KnotVariable = small_thing::Variable<Vec<4>>;

// The true path (YawDrift.h) of every rig for the current knot values, with what the derivatives
// of a relative transform H_A^-1 H_B with respect to the knots need. Psi_i and q_i are absolute
// (from the first rig); then for rigs A, B
//   d = Psi_B - Psi_A,   m = Rz(-Psi_A) (q_B - q_A)
// (YawDrifted's m and d, summed over any number of intervals), and
//   dd/dw_k = c_k(t_B) - c_k(t_A),
//   dm/dw_k = Rz(-Psi_A) (dq_B/dw_k - dq_A/dw_k) - c_k(t_A) z x m,
//   dm/du_k = Rz(-Psi_A) (dq_B/du_k - dq_A/du_k),
// c_k(t) = the part of knot interval k before t. dq_i/dw_k and dq_i/du_k are kept per knot as
// sums over the steps of its interval, plus, after it, L_k z x (q_i - q_end): O(rigs + knots).
class LoopPathCache {
 public:
  // rebuild if the knot values changed
  void update(const YawDriftPath& path, const std::vector<KnotVariable*>& knots);
  double psi(int i) const {
    return psi_[i];
  }
  const Vec3& q(int i) const {
    return q_[i];
  }
  // dq_i / dw_k and dq_i / du_k
  Vec3 dqdw(int i, int k) const;
  Mat33 dqdu(int i, int k) const;
  double coef(int k, double t) const; // c_k(t)

 private:
  const YawDriftPath* path_ = nullptr;
  std::vector<double> key_;
  std::vector<double> psi_;
  std::vector<Vec3> q_;
  // per knot: first rig index after which the interval's steps start contributing (a step j
  // contributes to knot k if it intersects its interval), the rig after its last such step, and
  // the sums over those steps
  std::vector<int> first_, end_;
  std::vector<Vec3> sumW_;
  std::vector<Mat33> sumU_;
  // per rig i: the partial sums for the (at most two) knots whose steps are still running at i
  struct Partial {
    int k;
    Vec3 w;
    Mat33 u;
  };
  std::vector<std::vector<Partial>> partial_;
};

struct LoopPointParam {
  small_thing::Variable<Vec3>* var = nullptr;
  small_thing::Variable<SE3>* host = nullptr;
  bool inverseDepth = false;
  Mat33 R0 = Mat33::Identity();
};

// a loss that scales the store's robust loss by the weight of a loop pair (virtual factors)
struct LoopPairLoss : small_thing::Loss {
  const small_thing::HuberLossWithCutoff* base = nullptr;
  const double* w = nullptr;
  double val(double s) const {
    return *w * base->val(s);
  }
  double der(double s) const {
    return *w * base->der(s);
  }
  std::pair<double, double> jet2(double s) const {
    const auto [v, d] = base->jet2(s);
    return {*w * v, *w * d};
  }
  std::tuple<double, double, double> jet3(double s) const {
    const auto [v, d, dd] = base->jet3(s);
    return {*w * v, *w * d, *w * dd};
  }
};

class LoopObservationStore : public small_thing::FactorStoreBase, public small_thing::MatrixFreeTerm {
 public:
  enum VarKind { kPoint, kPose, kCamera, kKnot };

  struct Observation {
    int point = -1; // index into points_
    small_thing::Variable<SE3>* rig = nullptr; // observing rig (== the host: host-rig observation)
    small_thing::Variable<SE3>* T_cam = nullptr;
    small_thing::Variable<CameraModelParam>* cam = nullptr;
    int rigA = -1, rigB = -1; // host and observer in the drift path (-1: no knots)
    int kLo = 0, kHi = -1; // knots between them (kHi < kLo: none)
    Vec2 px;
    Mat22 sqrtH;
    int group = 0, pair = 0;
    // variables (point, host, rig, T_cam, camera, knots kLo..kHi; host-rig observation: point,
    // T_cam, camera) and their kinds
    std::vector<small_thing::VarBase*> vars;
    std::vector<VarKind> kinds;
    mutable double prevCost = 0.0; // (-1: failed at the last linearization)
  };

  LoopObservationStore();

  // set up by SingleSessionProblem
  int addPoint(const LoopPointParam& p);
  void addObservation(Observation&& o);
  void setDrift(const YawDriftPath* path, const std::vector<KnotVariable*>& knots) {
    path_ = path;
    knots_ = knots;
  }
  // a preconditioner-only factor (store, index) of the virtual point `var` of a group of a point
  void addVirtualFactor(
      int point,
      small_thing::VarBase* var,
      small_thing::FactorStoreBase* store,
      int64_t k);
  LoopPairLoss& pairLoss(int pair);

  // robustness: the loss of every loop observation, and the weights of the pairs
  small_thing::HuberLossWithCutoff& loss() {
    return loss_;
  }
  void setPairWeight(int pair, double w);
  double pairWeight(int pair) const;
  int numPairs() const {
    return int(pairWeights_.size());
  }
  // median squared whitened error of the observations of every pair (no loss, no weight); -1 if
  // none could be evaluated
  std::vector<double> pairErrors() const;

  const std::vector<Observation>& observations() const {
    return obs_;
  }
  int numPoints() const {
    return int(points_.size());
  }
  // whitened error and Jacobian (2 x sum of the tangent dims of o.vars) of an observation
  bool evaluate(const Observation& o, Vec2& err, Mat<2, Eigen::Dynamic>* J) const;

  // FactorStoreBase
  int role() const override {
    return kRoleMatrixFree;
  }
  int64_t totalErrorDimensionality() const override {
    return 2 * int64_t(obs_.size());
  }
  int64_t numCosts() const override {
    return obs_.size();
  }
  bool unweightedError(int64_t i, Eigen::Ref<Eigen::VectorXd> err) const override;
  std::optional<double> varGradHess(
      int64_t i,
      int v,
      Eigen::Ref<Eigen::VectorXd> grad,
      Eigen::Ref<Eigen::MatrixXd> hess) const override;
  double unweightedSquaredError(int64_t i) const override;
  double covWeightedSquaredError(int64_t i) const override {
    return unweightedSquaredError(i);
  }
  std::optional<double> singleCost(int64_t i) const override;
  small_thing::VarBase* costVar(int64_t i, int v) const override {
    return v < int(obs_[i].vars.size()) ? obs_[i].vars[v] : nullptr;
  }
  std::string name() const override {
    return "visual_inertial_ba::LoopObservationStore";
  }
  bool verifyJacobians(
      double epsilon,
      double relativeTolerance,
      double absoluteTolerance,
      bool stopAtFirstError,
      const small_thing::LogFunc& log) const override;
  std::tuple<double, double, double> singleExpectedDelta(
      int64_t k,
      const double* stepData,
      const BaSpaCho::PermutedCoalescedAccessor& acc) const override;
  double computeCost(
      bool makeComparableWithStored,
      small_thing::CostStats* stats = nullptr,
      dispenso::ThreadPool* threadPool = nullptr) const override;
  double computeGradHess(
      double* gradData,
      const BaSpaCho::PermutedCoalescedAccessor& acc,
      double* hessData,
      bool updateCachedResults,
      bool dontRetryFailed,
      dispenso::ThreadPool* threadPool = nullptr) const override;
  void registerVariables(
      std::vector<int64_t>& paramSizes,
      small_thing::TypedStore<small_thing::VariableStoreBase>& varStores) override;
  void registerBlocks(std::unordered_set<std::pair<int64_t, int64_t>, small_thing::pair_hash>&)
      override {}

  // MatrixFreeTerm
  void linearize(const BaSpaCho::PermutedCoalescedAccessor& acc, double damping) override;
  void prepareRhs(double* vec) override;
  void correctReducedRhs(double* vec) override;
  void addMatvec(const double* v, double* out) override;
  void backSubstitute(double* vec) override;

  // (tests) after linearize: adds to a dense matrix in the solver's vector order the true
  // Gauss-Newton curvature of the loop observations, and to the points' blocks the part of their
  // damping the solver's diagonal does not have (it only has `damping` there: their blocks are
  // empty in the factorized matrix). Needs no virtual point.
  void addTrueHessianDense(Eigen::MatrixXd& H, double damping) const;

 private:
  // one linearized observation: whitened (weighted) error, the point's Jacobian, and the blocks of
  // the other (reduced) variables
  struct Lin {
    Vec2 r = Vec2::Zero();
    Mat23 Jp = Mat23::Zero();
    std::vector<std::pair<int64_t, int>> cols; // (full vector offset, dim) of reduced variables
    Mat<2, Eigen::Dynamic> Jx;
  };
  struct PointLin {
    std::vector<Lin> obs; // true observations
    Eigen::LDLT<Mat33> Hinv;
    Vec3 g = Vec3::Zero(); // gradient of the point (prepareRhs)
    int64_t slot = -1; // full vector offset of the point
  };
  struct VirtualLin {
    std::vector<Lin> obs;
    Eigen::LDLT<Mat33> Hinv;
  };
  struct VirtualFactor {
    small_thing::VarBase* var;
    small_thing::FactorStoreBase* store;
    int64_t k;
  };
  struct Point {
    LoopPointParam param;
    std::vector<int> obs;
    std::vector<std::vector<VirtualFactor>> virtualGroups;
  };
  double observationCost(const Observation& o, const Vec2& err) const;
  void addPointProduct(const std::vector<Lin>& obs, const Eigen::LDLT<Mat33>& Hinv, const double* v,
                       double* out, double sign) const;

  std::vector<Point> points_;
  std::vector<Observation> obs_;
  const YawDriftPath* path_ = nullptr;
  std::vector<KnotVariable*> knots_;
  mutable LoopPathCache cache_;
  small_thing::HuberLossWithCutoff loss_;
  std::deque<double> pairWeights_;
  std::deque<LoopPairLoss> pairLosses_;
  std::vector<PointLin> lin_;
  std::vector<std::vector<VirtualLin>> vlin_;
};

} // namespace visual_inertial_ba
