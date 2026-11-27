/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <camera_model/CameraModelParam.h>
#include <small_thing/Optimizer.h>
#include <viba/loop_closure/FivePoint.h>
#include <viba/problem/Types.h>

#include <deque>
#include <memory>
#include <optional>
#include <vector>

// Pose graph over the loop-closing key rigs (LC-KRs), 4 degrees of freedom per node.
//
// Node i keeps the roll and pitch of its open-loop pose: R_world_body(i) = Rz(psi_i) R0_i, where
// R0_i is the open-loop attitude and psi_i a yaw CORRECTION (0 at the start); its position p_i is
// free. The variable of a node is (psi_i, p_i). The world z axis is the vertical (as for VI-BA's
// drift knots). Node 0 is held (gauge).
//
// Stage 1: odometry edges + loop edges from two-view geometry (relative yaw, and the DIRECTION of
// the relative position of the two cameras: the scale is not used, see LoopEdgeFactor).
// Stage 2: odometry edges + loop points: every inlier match of a verified pair is a point, in
// inverse depth in the camera of the earlier rig, with reprojection errors in both rigs.
// Both: graduated non-convexity (GNC, Geman-McClure) over the loop pairs: one weight per pair,
// recomputed from the pair's residuals while the kernel is tightened.
// Two variants of the variables (PoseGraphVariant below): the nodes' 4-DoF poses, or drift knots
// only (the nodes then follow the open-loop trajectory through the knot transform).
namespace visual_inertial_ba::loop {

using NodeVar = small_thing::Variable<Vec4>; // (psi correction [rad], position [m])

struct CameraAtRig {
  const CameraModelParam* model = nullptr; // (not owned)
  SE3 T_cam_body;
};

struct LcNode {
  // (the description above is that of the RigPoses variant; see PoseGraphVariant)
  int64_t rigIndex = -1;
  int64_t timestampUs = 0;
  SE3 T_world_body; // open-loop pose (the true states, drift knots applied)
  std::vector<CameraAtRig> cameras; // by camera index
};

// A verified loop pair: camera camA of node A and camera camB of node B (A earlier in time)
struct LoopPair {
  int nodeA = -1, nodeB = -1;
  int camA = 0, camB = 0;
  RelativePose pose; // camera A -> camera B, |t| = 1
  bool directionValid = true; // enough parallax for the direction of t
  double medianParallaxRad = 0.0;
  double sigmaRotationRad = 0.0, sigmaDirectionRad = 0.0; // from the 5-point refinement
  std::vector<Vec2> pxA, pxB; // inlier correspondences (pixels)
  std::vector<Vec3> fA, fB; // their bearings
};

// yaw of a rotation that is close to a rotation about z
double yawOf(const Mat33& R);

// A loss with a weight (the GNC weight of its pair) on top of an optional Huber kernel
struct GncLoss : small_thing::Loss {
  double w = 1.0;
  double a = 0.0, b = 0.0; // Huber width a (<= 0: none), b = a^2
  explicit GncLoss(double huberWidth = 0.0) : a(huberWidth), b(huberWidth * huberWidth) {}
  double val(double s) const {
    return w * (a <= 0.0 || s <= b ? s : 2.0 * a * std::sqrt(s) - b);
  }
  double der(double s) const {
    return w * (a <= 0.0 || s <= b ? 1.0 : a / std::sqrt(s));
  }
  std::pair<double, double> jet2(double s) const {
    return {val(s), der(s)};
  }
  std::tuple<double, double, double> jet3(double s) const {
    if (a <= 0.0 || s <= b) {
      return {w * s, w, 0.0};
    }
    const double r = std::sqrt(s);
    return {w * (2.0 * a * r - b), w * a / r, -w * a / (2.0 * s * r)};
  }
};

// odometry edge: r = [(psi_j - psi_i) / s_yaw; (Rz(-psi_i) (p_j - p_i) - dp0) ./ s_pos]
struct OdometryFactor {
  OdometryFactor(const Vec3& dp0, double sigmaYaw, const Vec3& sigmaPos)
      : dp0_(dp0), wYaw_(1.0 / sigmaYaw), wPos_(sigmaPos.cwiseInverse()) {}
  Vec4 operator()(const Vec4& xi, const Vec4& xj, Ref<Mat<4, 4>> Ji, Ref<Mat<4, 4>> Jj) const;

  Vec3 dp0_;
  double wYaw_;
  Vec3 wPos_;
};

// stage-1 loop edge between nodes i (camera a) and j (camera b): the yaw of the measured relative
// rotation, and the direction of the camera centre of j seen from the camera of i (2 components
// in the tangent plane of the measured direction; zero if the direction is not valid). Only the
// direction of the 5-point translation is used: its length is unknown and is NOT taken from the
// stereo pair.
struct LoopEdgeFactor {
  LoopEdgeFactor(
      const Mat33& R0i,
      const Mat33& R0j,
      const SE3& T_ca_body,
      const SE3& T_cb_body,
      const RelativePose& meas,
      bool directionValid,
      double sigmaYaw,
      double sigmaDir);
  Vec3 operator()(const Vec4& xi, const Vec4& xj, Ref<Mat<3, 4>> Ji, Ref<Mat<3, 4>> Jj) const;

  Mat33 R0i_, R0j_;
  Mat33 R_ca_body_; // body -> camera a
  Vec3 la_, lb_; // camera centres in the bodies
  double yawMeas_; // yaw of R0i R_body_ca R^T R_body_cb^T R0j^T
  Vec3 b1_, b2_; // tangent basis at the measured direction
  bool directionValid_;
  double wYaw_, wDir_;
};

// stage-2 point (u, v, rho): position R0 [u, v, 1] / rho in camera a of node i (the host)
// observation from the host camera itself
struct LoopPointHostObsFactor {
  LoopPointHostObsFactor(
      const CameraModelParam* cam,
      const Mat33& R0,
      const Vec2& px,
      double sqrtInfo)
      : cam_(cam), R0_(R0), px_(px), sqrtH_(Mat22::Identity() * sqrtInfo) {}
  std::optional<Vec2> operator()(const Vec3& pt, Ref<Mat23> Jpt) const;

  const CameraModelParam* cam_;
  Mat33 R0_;
  Vec2 px_;
  Mat22 sqrtH_;
};

// observation of the point from camera b of node j
struct LoopPointObsFactor {
  LoopPointObsFactor(
      const CameraModelParam* camB,
      const Mat33& R0,
      const Mat33& R0i,
      const Mat33& R0j,
      const SE3& T_ca_body,
      const SE3& T_cb_body,
      const Vec2& px,
      double sqrtInfo);
  std::optional<Vec2> operator()(
      const Vec3& pt,
      const Vec4& xi,
      const Vec4& xj,
      Ref<Mat23> Jpt,
      Ref<Mat<2, 4>> Ji,
      Ref<Mat<2, 4>> Jj) const;

  const CameraModelParam* camB_;
  Mat33 R0_; // bearing frame of the point in camera a
  Mat33 R0i_, R0j_;
  Mat33 R_body_ca_, R_cb_body_;
  Vec3 la_, lb_;
  Vec2 px_;
  Mat22 sqrtH_;
};

struct GncSettings {
  bool enabled = true;
  double thresholdSq = 9.0; // c^2: a pair is an inlier if the median squared whitened error of its
                            // factors is below
  double muFactor = 1.4; // the control parameter is divided by this at every round
  int innerIterations = 8; // optimizer iterations per round
  int finalIterations = 30;
};

// The two variants of the pose graph (user, 2026-11-26), selected by an option:
//   RigPoses: the variables are the nodes' 4-DoF poses (psi_i, p_i); node 0 is held.
//   Knots:    the variables are drift knots only (yaw rate w and drift velocity u, constant on each
//             knot interval, YawDrift.h); the nodes' poses are FUNCTIONS of the knots: the
//             open-loop trajectory (all key rigs, held) put through the knot transform, so node i
//             is (Psi(t_i), q_i). No global pose variable. The gauge is the start of the recording
//             (Psi = 0, q = ref there) plus weak priors on the knots.
// The factors are the same in both: in the Knots variant they are evaluated on the node states
// computed from the knots, and their Jacobians are chained through d(node state) / d(knots).
enum class PoseGraphVariant { RigPoses, Knots };

// what the Knots variant needs: every key rig of the open-loop trajectory and the knot times
struct KnotLayout {
  std::vector<int64_t> rigTimesUs; // all key rigs, time order
  std::vector<Vec3> rigPos; // their open-loop IMU positions (the frozen copy)
  std::vector<double> knotTimeSec; // numKnots + 1 times, from the first rig
  std::vector<int> nodeRig; // node -> index into rigTimesUs (increasing)
  double rateSigma = 0.01; // knot priors at 0 [rad/s]
  double velSigma = 1.0; // [m/s]
};

// node states (psi, position) and their derivatives with respect to the knots, for the current
// value of the knot vector (recomputed when it changes; single-threaded use)
class KnotNodeStates {
 public:
  explicit KnotNodeStates(const KnotLayout& layout);
  int numKnots() const {
    return int(layout_.knotTimeSec.size()) - 1;
  }
  // node state and its Jacobian (4 x 4 numKnots) for the knot vector `knots`
  const Vec4& state(const VecX& knots, int node) const;
  const Mat<4, Eigen::Dynamic>& jacobian(const VecX& knots, int node) const;
  const KnotLayout& layout() const {
    return layout_;
  }

 private:
  void update(const VecX& knots) const;
  KnotLayout layout_;
  mutable VecX cachedKnots_;
  mutable std::vector<Vec4> states_;
  mutable std::vector<Mat<4, Eigen::Dynamic>> jacobians_;
};

// a factor between two nodes (OdometryFactor, LoopEdgeFactor) on the knot vector
template <class Base, int E>
struct KnotPairFactor {
  KnotPairFactor(Base base, std::shared_ptr<const KnotNodeStates> states, int i, int j)
      : base_(std::move(base)), states_(std::move(states)), i_(i), j_(j) {}
  Vec<E> operator()(const VecX& knots, Ref<Mat<E, Eigen::Dynamic>> J) const {
    Mat<E, 4> Ji, Jj;
    const bool needJ = !small_thing::isNull(J);
    const Vec<E> e = needJ
        ? base_(states_->state(knots, i_), states_->state(knots, j_), Ji, Jj)
        : base_(
              states_->state(knots, i_),
              states_->state(knots, j_),
              small_thing::NullRef(),
              small_thing::NullRef());
    if (needJ) {
      J = Ji * states_->jacobian(knots, i_) + Jj * states_->jacobian(knots, j_);
    }
    return e;
  }
  Base base_;
  std::shared_ptr<const KnotNodeStates> states_;
  int i_, j_;
};

// LoopPointObsFactor on (point, knot vector)
struct KnotPointObsFactor {
  KnotPointObsFactor(
      LoopPointObsFactor base,
      std::shared_ptr<const KnotNodeStates> states,
      int i,
      int j)
      : base_(std::move(base)), states_(std::move(states)), i_(i), j_(j) {}
  std::optional<Vec2> operator()(
      const Vec3& pt,
      const VecX& knots,
      Ref<Mat23> Jpt,
      Ref<Mat<2, Eigen::Dynamic>> J) const;
  LoopPointObsFactor base_;
  std::shared_ptr<const KnotNodeStates> states_;
  int i_, j_;
};

// prior at 0 on knot k of the knot vector
struct KnotPriorFactor {
  KnotPriorFactor(int k, int numKnots, double rateSigma, double velSigma)
      : k_(k), n_(numKnots), w_(1.0 / rateSigma, 1.0 / velSigma, 1.0 / velSigma, 1.0 / velSigma) {}
  Vec4 operator()(const VecX& knots, Ref<Mat<4, Eigen::Dynamic>> J) const {
    if (!small_thing::isNull(J)) {
      J.setZero();
      J.block<4, 4>(0, 4 * k_) = w_.asDiagonal();
    }
    return w_.cwiseProduct(knots.segment<4>(4 * k_));
  }
  int k_, n_;
  Vec4 w_;
};

class PoseGraph {
 public:
  // RigPoses variant
  explicit PoseGraph(const std::vector<LcNode>& nodes);
  // Knots variant (layout.nodeRig has one entry per node)
  PoseGraph(const std::vector<LcNode>& nodes, const KnotLayout& layout);

  PoseGraphVariant variant() const {
    return knotStates_ ? PoseGraphVariant::Knots : PoseGraphVariant::RigPoses;
  }

  // start values: RigPoses: yaw corrections and positions of the nodes (default: 0 and the
  // open-loop positions); Knots: setStartKnots
  void setStart(const std::vector<double>& yaw, const std::vector<Vec3>& pos);
  void setStartKnots(const VecX& knots);

  void addOdometry(int i, int j, double sigmaYaw, const Vec3& sigmaPos);

  // stage 1: one edge per pair; returns the pair's group
  int addLoopEdge(const LoopPair& pair, double sigmaYaw, double sigmaDir);

  // stage 2: one point per inlier of the pair, triangulated from the current node values;
  // returns the pair's group (-1 if no point could be made). All points before the first solve.
  int addLoopPoints(const LoopPair& pair, double pixelSigma, double huberPx);

  // GNC over the groups, then the final solve with the accepted groups at weight 1 and the
  // others at 0; returns false if the optimizer failed
  bool solve(const GncSettings& gnc, int numThreads = 1);

  int numNodes() const {
    return int(nodes_.size());
  }
  // node state: (yaw correction, position)
  Vec4 nodeState(int i) const;
  double yaw(int i) const {
    return nodeState(i)[0];
  }
  Vec3 position(int i) const {
    return nodeState(i).tail<3>();
  }
  // corrected pose of a node
  SE3 T_world_body(int i) const;
  // Knots variant: the knot vector (w, u) x numKnots
  const VecX& knots() const {
    return knotVar_.value;
  }
  int numGroups() const {
    return int(groups_.size());
  }
  double groupWeight(int g) const {
    return groups_[g].loss->w;
  }
  // median squared whitened error of a group's factors (without weight and kernel)
  double groupError(int g) const;
  int numPoints() const {
    return int(points_.size());
  }
  small_thing::Optimizer& optimizer() {
    return opt_;
  }
  // total cost (weights and kernels included)
  double cost() const {
    return opt_.computeCost();
  }

 private:
  struct Group {
    std::unique_ptr<GncLoss> loss;
    std::vector<std::pair<small_thing::FactorStoreBase*, int64_t>> factors;
  };
  void registerOnce();

  std::vector<LcNode> nodes_;
  std::vector<NodeVar> vars_; // RigPoses variant
  small_thing::Variable<VecX> knotVar_{VecX()}; // Knots variant
  std::shared_ptr<KnotNodeStates> knotStates_; // Knots variant
  std::deque<small_thing::Variable<Vec3>> points_;
  std::vector<Group> groups_;
  small_thing::Optimizer opt_;
  bool registered_ = false;
};

} // namespace visual_inertial_ba::loop
