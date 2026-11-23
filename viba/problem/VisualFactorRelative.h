/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <camera_model/CameraModelParam.h>
#include <small_thing/Common.h>
#include <viba/problem/Types.h>

#include <optional>

// Visual factors for points expressed in a host rig instead of the world.
//
// The variable of the point is either its position in the body frame of the host rig ("cartesian"),
// or (u, v, rho) with the position R0 [u, v, 1] / rho, where R0 is a fixed rotation whose third
// axis is the initial direction of the point in the host ("inverse depth"). The host is one of the
// rigs observing the point.
//
// Why: with world coordinates a joint motion of a stretch of trajectory and of the points it sees
// (e.g. a slow change of heading along a recording) has to move every point, so the
// Levenberg-Marquardt damping of the points resists it; and a low-parallax point, which the
// optimizer pushes ever farther, would have to move sideways by its distance times the rotation,
// which the damping does not allow: it then pins the absolute heading of the cameras that see it.
// Expressed in a host rig the point goes with its host, and a point at infinity (rho = 0) is an
// ordinary value.
//
// All the geometry is done on w * point (w = 1 or rho), which is only normalized before the
// projection. rho <= 0 is not an error: w * point stays a valid direction through rho = 0 (the
// point passes through infinity); the observation fails only when that direction is behind the
// camera.
//
// NOTE: the names start with "VisualFactor" on purpose: the factor stores are found by name (point
// refinement, histograms). The point is variable 0, as in the other visual factors.

namespace visual_inertial_ba {

namespace relpoint {
inline void homogeneous(bool inverseDepth, const Mat33& R0, const Vec3& pt, Vec3& m, double& w) {
  if (inverseDepth) {
    m = R0 * Vec3(pt[0], pt[1], 1.0);
    w = pt[2];
  } else {
    m = pt;
    w = 1.0;
  }
}

// q: w * (point in the camera), w > 0 -> whitened error, A = d whiteErr / d q, and the camera-model
// Jacobian
inline bool projectScaled(
    const CameraModelParam& cameraModel,
    const Vec3& q,
    const Vec2& projBaseRes,
    const Mat22& sqrtH_BaseRes,
    Vec2& whiteErr,
    Mat23& A,
    Mat<2, Eigen::Dynamic>& dWhiteErr_dCamParams) {
  const double n = q.norm();
  if (!(n > 0.0)) {
    return false;
  }
  const Vec3 qh = q / n;
  Vec2 proj;
  Mat23 dProj_dPointCam;
  Mat<2, Eigen::Dynamic> dProj_dCamParams(2, cameraModel.numModelParameters());
  if (!cameraModel.project(qh, proj, dProj_dPointCam, dProj_dCamParams)) {
    return false;
  }
  whiteErr = sqrtH_BaseRes * (proj - projBaseRes);
  A = sqrtH_BaseRes * dProj_dPointCam * (Mat33::Identity() - qh * qh.transpose()) / n;
  dWhiteErr_dCamParams = sqrtH_BaseRes * dProj_dCamParams;
  return true;
}
} // namespace relpoint

// observation from a rig other than the host
struct VisualFactorHosted {
  VisualFactorHosted(
      const Vec2& projBaseRes,
      const Mat22& sqrtH_BaseRes,
      bool inverseDepth,
      const Mat33& R0)
      : projBaseRes_(projBaseRes),
        sqrtH_BaseRes_(sqrtH_BaseRes),
        inverseDepth_(inverseDepth),
        R0_(R0) {}

  std::optional<Vec2> operator()(
      const Vec3& pt,
      const SE3& T_hostBodyImu_world,
      const SE3& T_bodyImu_world,
      const SE3& T_Cam_BodyImu,
      const CameraModelParam& cameraModel,
      Ref<Mat23> pt_Jacobian,
      Ref<Mat26> T_hostBodyImu_world_Jacobian,
      Ref<Mat26> T_bodyImu_world_Jacobian,
      Ref<Mat26> T_Cam_BodyImu_Jacobian,
      Ref<Mat<2, Eigen::Dynamic>> cameraModel_Jacobian) const {
    Vec3 m;
    double w;
    relpoint::homogeneous(inverseDepth_, R0_, pt, m, w);
    const Mat33 Rh = T_hostBodyImu_world.so3().matrix();
    const Mat33 Ro = T_bodyImu_world.so3().matrix();
    const Mat33 Rc = T_Cam_BodyImu.so3().matrix();
    const Vec3 g = Rh.transpose() * (m - w * T_hostBodyImu_world.translation()); // w * world point
    const Vec3 s = Ro * g + w * T_bodyImu_world.translation(); // w * point in the observing body
    const Vec3 q = Rc * s + w * T_Cam_BodyImu.translation(); // w * point in the camera
    Vec2 whiteErr;
    Mat23 A;
    Mat<2, Eigen::Dynamic> dCam;
    if (!relpoint::projectScaled(cameraModel, q, projBaseRes_, sqrtH_BaseRes_, whiteErr, A, dCam)) {
      return {};
    }
    const Mat33 M = Rc * Ro * Rh.transpose();
    const Mat23 AM = A * M;
    if (!small_thing::isNull(pt_Jacobian)) {
      if (inverseDepth_) {
        const Vec3 t_cam_host =
            Rc * (Ro * (-(Rh.transpose() * T_hostBodyImu_world.translation())) +
                  T_bodyImu_world.translation()) +
            T_Cam_BodyImu.translation();
        Mat33 D;
        D.col(0) = M * R0_.col(0);
        D.col(1) = M * R0_.col(1);
        D.col(2) = t_cam_host;
        pt_Jacobian = A * D;
      } else {
        pt_Jacobian = AM;
      }
    }
    if (!small_thing::isNull(T_hostBodyImu_world_Jacobian)) {
      T_hostBodyImu_world_Jacobian << -w * AM, AM * SO3::hat(m);
    }
    if (!small_thing::isNull(T_bodyImu_world_Jacobian)) {
      const Mat23 AR = A * Rc;
      T_bodyImu_world_Jacobian << w * AR, AR * SO3::hat(-s);
    }
    if (!small_thing::isNull(T_Cam_BodyImu_Jacobian)) {
      T_Cam_BodyImu_Jacobian << w * A, A * SO3::hat(-q);
    }
    if (!small_thing::isNull(cameraModel_Jacobian)) {
      cameraModel_Jacobian.leftCols(cameraModel.numModelParameters()) = dCam;
    }
    return whiteErr;
  }

  double imageRow() const {
    return projBaseRes_[1];
  }

 private:
  Vec2 projBaseRes_;
  Mat22 sqrtH_BaseRes_;
  bool inverseDepth_;
  Mat33 R0_;
};

// observation from the host rig itself
struct VisualFactorHostRig {
  VisualFactorHostRig(
      const Vec2& projBaseRes,
      const Mat22& sqrtH_BaseRes,
      bool inverseDepth,
      const Mat33& R0)
      : projBaseRes_(projBaseRes),
        sqrtH_BaseRes_(sqrtH_BaseRes),
        inverseDepth_(inverseDepth),
        R0_(R0) {}

  std::optional<Vec2> operator()(
      const Vec3& pt,
      const SE3& T_Cam_BodyImu,
      const CameraModelParam& cameraModel,
      Ref<Mat23> pt_Jacobian,
      Ref<Mat26> T_Cam_BodyImu_Jacobian,
      Ref<Mat<2, Eigen::Dynamic>> cameraModel_Jacobian) const {
    Vec3 m;
    double w;
    relpoint::homogeneous(inverseDepth_, R0_, pt, m, w);
    const Mat33 Rc = T_Cam_BodyImu.so3().matrix();
    const Vec3 q = Rc * m + w * T_Cam_BodyImu.translation();
    Vec2 whiteErr;
    Mat23 A;
    Mat<2, Eigen::Dynamic> dCam;
    if (!relpoint::projectScaled(cameraModel, q, projBaseRes_, sqrtH_BaseRes_, whiteErr, A, dCam)) {
      return {};
    }
    if (!small_thing::isNull(pt_Jacobian)) {
      if (inverseDepth_) {
        Mat33 D;
        D.col(0) = Rc * R0_.col(0);
        D.col(1) = Rc * R0_.col(1);
        D.col(2) = T_Cam_BodyImu.translation();
        pt_Jacobian = A * D;
      } else {
        pt_Jacobian = A * Rc;
      }
    }
    if (!small_thing::isNull(T_Cam_BodyImu_Jacobian)) {
      T_Cam_BodyImu_Jacobian << w * A, A * SO3::hat(-q);
    }
    if (!small_thing::isNull(cameraModel_Jacobian)) {
      cameraModel_Jacobian.leftCols(cameraModel.numModelParameters()) = dCam;
    }
    return whiteErr;
  }

  double imageRow() const {
    return projBaseRes_[1];
  }

 private:
  Vec2 projBaseRes_;
  Mat22 sqrtH_BaseRes_;
  bool inverseDepth_;
  Mat33 R0_;
};

} // namespace visual_inertial_ba
