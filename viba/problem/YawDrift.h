/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the Apache-2.0 license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <small_thing/Common.h>
#include <viba/problem/Types.h>

#include <algorithm>
#include <cmath>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

namespace visual_inertial_ba {

// Yaw-drift knots: the rate w(t) [rad/s] at which the estimated trajectory turns about the
// vertical with respect to the true one, CONSTANT between knot times (knot k is the rate on
// [knotTime[k], knotTime[k + 1])). Psi(t) is its integral from the first rig: exact, additive.
//
// The true states are a transform of the variables that depends on the time and on the knots
// only (a gauge: for every value of the knots there are variables with the same true states, so
// the knots over-parametrize the problem without changing it):
//   R_world_rig(true) = Rz(Psi_i) * R_world_rig        vel(true) = Rz(Psi_i) * vel
//   p(true)           = q_i + Rz(Psi_i) * (p - ref_i)
// where ref_i is a frozen copy of the positions (YawDriftPath::refPos, taken when the knots are
// 0) and q is that copy with the drift applied progressively from the start:
//   q_0 = ref_0,   q_{i+1} = q_i + Rz((Psi_i + Psi_{i+1}) / 2) * (ref_{i+1} - ref_i)
// So with the variables at the frozen copy, a knot turns the steps of the trajectory, each by the
// drift accumulated up to it, and what the variables differ from the copy by is turned with the
// rig.
//
// The factors that relate two rigs A and B (two consecutive rigs of an inertial factor; the host
// rig of a point and the rig that observes it) do not change when both rigs are turned about the
// vertical and shifted together, so they are evaluated with A as it is and B taken by
// H_A^-1 H_B (H_i: the transform above at rig i):
//   R_world_B -> Rz(d) * R_world_B,   vel_B -> Rz(d) * vel_B,      d = Psi_B - Psi_A
//   p_B -> ref_A + m + Rz(d) * (p_B - ref_B),   m = Rz(-Psi_A) * (q_B - q_A)
//     = sum over the steps j between A and B of
//       Rz((Psi_j + Psi_{j+1}) / 2 - Psi_A) * (ref_{j+1} - ref_j)
// d and m depend on the knots between the two times only: two of them if the rigs are in the
// same or in neighbouring knot intervals (YawDriftPath::knotPairFor), which is what is handled.
// The knots are put into the variables often (SingleSessionProblem::yawKnots_apply: variables =
// true states, knots = 0, frozen copy taken again; before every optimizer iteration): left to
// accumulate they take the variables far from the copy and the problem becomes very non-linear.
//
// Shift (YawDriftPath::shift): a knot is 4 numbers, the rate of turning and a velocity u [m/s] at
// which the estimated positions drift in the estimated world, constant on the knot's interval.
// It is added to the steps before they are turned:
//   q_{i+1} = q_i + Rz((Psi_i + Psi_{i+1}) / 2) * (ref_{i+1} - ref_i + U_i),  U_i = integral of u
// so that m has the steps with their U. The velocities are turned only (not u added): a factor
// of one rig alone that uses its attitude and velocity (rolling shutter) then sees the same
// gravity and velocity in the body whatever the knots, and needs no wrapping.
//
//   Base: the wrapped factor, E: size of its error, kPoseA / kVelA / kPoseB / kVelB: positions of
//   T_rig_world and vel_world of the two rigs among its variables (velocities: -1 if not there),
//   V...: the value types of its variables. The two knots are appended as the last two variables.
// Perturbations: T_rig_world x = R_rig_world * (x - p); a left perturbation (dt, dw) of T moves p
// by -R_world_rig * dt and turns R_rig_world by exp(dw). The pose given to the factor for B is
// T_B composed with a transform of the world, so its left perturbations are those of T_B; a
// change dd of d and dp of its position are the left perturbation (-R'_rig_world * dp,
// -R'_rig_world * z * dd).
// (the wrapped factor's name is part of this one's, which is how the factor stores are found)
struct YawDriftPath {
  std::vector<double> timeSec; // of every rig, in time order, from the first rig
  std::vector<Vec3> refPos; // frozen positions of the IMU in the world, same order
  std::vector<double> knotTimeSec; // numKnots + 1 times: knot k is the rate on [k, k + 1)
  bool active = true; // false: the plain factors
  bool shift = false; // the knots' velocities are used (otherwise they stay 0)

  int numKnots() const {
    return int(knotTimeSec.size()) - 1;
  }
  int intervalOf(double t) const {
    const int k =
        int(std::upper_bound(knotTimeSec.begin(), knotTimeSec.end(), t) - knotTimeSec.begin()) - 1;
    return std::min(std::max(k, 0), numKnots() - 1);
  }
  // the first of the two knots that cover the rigs a and b (indices in time order); -1 if the
  // rigs are more than one knot interval apart
  int knotPairFor(int a, int b) const {
    const int ka = intervalOf(timeSec[std::min(a, b)]);
    const int kb = intervalOf(timeSec[std::max(a, b)]);
    return kb - ka > 1 ? -1 : std::min(ka, numKnots() - 2);
  }
  // Psi(t) - Psi(knotTimeSec[k0]) = a * knot[k0] + b * knot[k0 + 1], t in these two intervals
  void coef(int k0, double t, double& a, double& b) const {
    const double t1 = knotTimeSec[k0 + 1];
    a = std::min(t, t1) - knotTimeSec[k0];
    b = std::max(t - t1, 0.0);
  }
  // integral of the knots' velocities between two times, for velocities given by a function of
  // the knot index
  template <class KnotVelocity>
  Vec3 shiftBetween(double t0, double t1, const KnotVelocity& velocity) const {
    Vec3 sum = Vec3::Zero();
    for (int k = intervalOf(t0); k < numKnots() && knotTimeSec[k] < t1; k++) {
      const double overlap = std::min(t1, knotTimeSec[k + 1]) - std::max(t0, knotTimeSec[k]);
      if (overlap > 0.0) {
        sum += overlap * velocity(k);
      }
    }
    return sum;
  }
  // Psi(t) for knot values given by a function of the knot index
  template <class KnotValue>
  double psiAt(double t, const KnotValue& knot) const {
    double psi = 0.0;
    for (int k = 0; k < numKnots() && knotTimeSec[k] < t; k++) {
      psi += knot(k) * (std::min(t, knotTimeSec[k + 1]) - knotTimeSec[k]);
    }
    return psi;
  }
};

namespace yawdrift {
template <int E, class V>
struct JacobianOf {
  using type = Mat<E, Eigen::Dynamic>; // calibration blocks of dynamic size
};
template <int E>
struct JacobianOf<E, Vec3> {
  using type = Mat<E, 3>;
};
template <int E>
struct JacobianOf<E, SE3> {
  using type = Mat<E, 6>;
};
template <int E>
struct JacobianOf<E, Vec<1>> {
  using type = Mat<E, 1>;
};
template <int E>
struct JacobianOf<E, Vec<4>> {
  using type = Mat<E, 4>;
};

template <class T>
bool hasValue(const std::optional<T>& v) {
  return v.has_value();
}
template <class T>
bool hasValue(const T&) {
  return true;
}
} // namespace yawdrift

template <class Base, int E, int kPoseA, int kVelA, int kPoseB, int kVelB, class... V>
struct YawDrifted {
  using Knot = Vec<4>; // rate of turning [rad/s], velocity of the shift [m/s]
  using KnotJacobian = Mat<E, 4>;

  // rigA, rigB: indices of the two rigs in path (time order); knot0 = path->knotPairFor(rigA,
  // rigB) >= 0: the factor's two knots are knot0 and knot0 + 1
  YawDrifted(const Base& base, const YawDriftPath* path, int rigA, int rigB, int knot0)
      : base_(base), path_(path), rigA_(rigA), rigB_(rigB), knot0_(knot0) {}

  auto operator()(
      const V&... values,
      const Knot& knotA,
      const Knot& knotB,
      Ref<typename yawdrift::JacobianOf<E, V>::type>... jacobians,
      Ref<KnotJacobian> knotA_Jacobian,
      Ref<KnotJacobian> knotB_Jacobian) const {
    return call(
        std::index_sequence_for<V...>{},
        std::forward_as_tuple(values...),
        std::forward_as_tuple(jacobians...),
        knotA,
        knotB,
        knotA_Jacobian,
        knotB_Jacobian);
  }

 private:
  template <int I, class O, class PB, class VB>
  static decltype(auto) sel(O&& o, PB&& pb, VB&& vb) {
    if constexpr (I == kPoseB) {
      return std::forward<PB>(pb);
    } else if constexpr (I == kVelB) {
      return std::forward<VB>(vb);
    } else {
      return std::forward<O>(o);
    }
  }

  template <int K, class Tuple>
  static decltype(auto) getOr(Tuple&& t, const Vec3& fallback) {
    if constexpr (K >= 0) {
      return std::get<K>(t);
    } else {
      return (fallback);
    }
  }

  template <size_t... Is, class Values, class Jacobians>
  auto call(
      std::index_sequence<Is...>,
      Values&& values,
      Jacobians&& jacobians,
      const Knot& knotA,
      const Knot& knotB,
      Ref<KnotJacobian> knotA_Jacobian,
      Ref<KnotJacobian> knotB_Jacobian) const {
    using PoseJacobian = Mat<E, 6>;
    using VelJacobian = Mat<E, 3>;
    const YawDriftPath& path = *path_;
    if (!path.active) {
      auto maybeErr = base_(std::get<Is>(values)..., std::get<Is>(jacobians)...);
      if (yawdrift::hasValue(maybeErr)) {
        if (!small_thing::isNull(knotA_Jacobian)) {
          knotA_Jacobian.setZero();
        }
        if (!small_thing::isNull(knotB_Jacobian)) {
          knotB_Jacobian.setZero();
        }
      }
      return maybeErr;
    }
    const Vec3 zero = Vec3::Zero();
    const Vec3 up = Vec3::UnitZ();

    // d = Psi_B - Psi_A = w[0] * cd[0] + w[1] * cd[1]
    const double w[2] = {knotA[0], knotB[0]};
    const Vec3 u[2] = {
        path.shift ? Vec3(knotA.template tail<3>()) : zero,
        path.shift ? Vec3(knotB.template tail<3>()) : zero};
    double aA, bA, aB, bB;
    path.coef(knot0_, path.timeSec[rigA_], aA, bA);
    path.coef(knot0_, path.timeSec[rigB_], aB, bB);
    const double cd[2] = {aB - aA, bB - bA};
    const double d = w[0] * cd[0] + w[1] * cd[1];

    // m = Rz(-Psi_A) * (q_B - q_A), and its derivatives with respect to the two knots (rates:
    // dm; velocities: the matrices [cM -sM 0; sM cM 0; 0 0 zM])
    const int lo = std::min(rigA_, rigB_), hi = std::max(rigA_, rigB_);
    const double sgn = rigA_ <= rigB_ ? 1.0 : -1.0;
    Vec3 m = Vec3::Zero(), dm[2] = {Vec3::Zero(), Vec3::Zero()};
    double cM[2] = {0.0, 0.0}, sM[2] = {0.0, 0.0}, zM[2] = {0.0, 0.0};
    double aPrev, bPrev;
    path.coef(knot0_, path.timeSec[lo], aPrev, bPrev);
    for (int j = lo; j < hi; j++) {
      double aNext, bNext;
      path.coef(knot0_, path.timeSec[j + 1], aNext, bNext);
      const double am = 0.5 * (aPrev + aNext) - aA, bm = 0.5 * (bPrev + bNext) - bA;
      const double theta = w[0] * am + w[1] * bm;
      const double c = std::cos(theta), s = std::sin(theta);
      // (the part of the step in each of the two intervals)
      const double overlap[2] = {aNext - aPrev, bNext - bPrev};
      const Vec3 step =
          path.refPos[j + 1] - path.refPos[j] + overlap[0] * u[0] + overlap[1] * u[1];
      for (int n = 0; n < 2; n++) {
        cM[n] += sgn * overlap[n] * c;
        sM[n] += sgn * overlap[n] * s;
        zM[n] += sgn * overlap[n];
      }
      const Vec3 r(c * step.x() - s * step.y(), s * step.x() + c * step.y(), step.z());
      const Vec3 zxr(-r.y(), r.x(), 0.0);
      m += sgn * r;
      dm[0] += (sgn * am) * zxr;
      dm[1] += (sgn * bm) * zxr;
      aPrev = aNext;
      bPrev = bNext;
    }

    // rig B, taken by H_A^-1 H_B
    const SE3& T_B = std::get<kPoseB>(values);
    const SO3 Rz = SO3::rotZ(d);
    const Vec3 pos_B = -(T_B.so3().inverse() * T_B.translation());
    const Vec3 lever = Rz * (pos_B - path.refPos[rigB_]);
    const Vec3 moved_pos_B = path.refPos[rigA_] + m + lever;
    const SO3 moved_R_B = T_B.so3() * Rz.inverse();
    const SE3 moved_T_B(moved_R_B, -(moved_R_B * moved_pos_B));
    const Vec3 moved_vel_B = Rz * getOr<kVelB>(values, zero);

    const bool needKnots =
        !small_thing::isNull(knotA_Jacobian) || !small_thing::isNull(knotB_Jacobian);
    PoseJacobian jB;
    VelJacobian jvB = VelJacobian::Zero();
    Ref<PoseJacobian> jBRef =
        needKnots || !small_thing::isNull(std::get<kPoseB>(jacobians))
        ? Ref<PoseJacobian>(jB)
        : small_thing::NullRef();
    bool needVelB = false;
    if constexpr (kVelB >= 0) {
      needVelB = needKnots || !small_thing::isNull(std::get<kVelB>(jacobians));
    }
    Ref<VelJacobian> jvBRef = needVelB ? Ref<VelJacobian>(jvB) : small_thing::NullRef();

    auto maybeErr = base_(
        sel<int(Is)>(std::get<Is>(values), moved_T_B, moved_vel_B)...,
        sel<int(Is)>(std::get<Is>(jacobians), jBRef, jvBRef)...);
    if (!yawdrift::hasValue(maybeErr)) {
      return maybeErr;
    }

    if (!small_thing::isNull(std::get<kPoseB>(jacobians))) {
      std::get<kPoseB>(jacobians) = jB;
    }
    if constexpr (kVelB >= 0) {
      if (!small_thing::isNull(std::get<kVelB>(jacobians))) {
        std::get<kVelB>(jacobians) = jvB * Rz.matrix();
      }
    }
    if (needKnots) {
      const Vec3 upInB = moved_R_B * up;
      const Vec3 dVel = up.cross(moved_vel_B);
      const Vec3 dLever = up.cross(lever);
      Ref<KnotJacobian>* out[2] = {&knotA_Jacobian, &knotB_Jacobian};
      for (int n = 0; n < 2; n++) {
        if (!small_thing::isNull(*out[n])) {
          const Vec3 dPos = dm[n] + cd[n] * dLever;
          out[n]->col(0) = -(jB.template leftCols<3>() * (moved_R_B * dPos)) -
              cd[n] * (jB.template rightCols<3>() * upInB) + cd[n] * (jvB * dVel);
          if (path.shift) {
            Mat33 dm_du;
            dm_du << cM[n], -sM[n], 0.0, sM[n], cM[n], 0.0, 0.0, 0.0, zM[n];
            out[n]->template rightCols<3>() =
                -(jB.template leftCols<3>() * (moved_R_B.matrix() * dm_du));
          } else {
            out[n]->template rightCols<3>().setZero();
          }
        }
      }
    }
    return maybeErr;
  }

  Base base_;
  const YawDriftPath* path_;
  int rigA_, rigB_, knot0_;
};

// YawDrifted for the variables `vars...` (their value types are taken from them)
template <class Base, int E, int kPoseA, int kVelA, int kPoseB, int kVelB, class... Vars>
using YawDriftedFor = YawDrifted<
    Base,
    E,
    kPoseA,
    kVelA,
    kPoseB,
    kVelB,
    std::decay_t<decltype(std::declval<Vars&>().value)>...>;

} // namespace visual_inertial_ba
