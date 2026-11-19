/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>

#include <calibration/projections/Common.h>

namespace calibration {

// Pinhole camera with radial-tangential (Brown-Conrady) distortion, as in OpenCV and in Kalibr's
// "pinhole-radtan" (EuRoC). With a = x/z, b = y/z, r2 = a^2 + b^2:
//
//   radial = 1 + k1 r2 + k2 r2^2 (+ k3 r2^3)
//   xd = a radial + 2 p1 a b + p2 (r2 + 2 a^2)
//   yd = b radial + p1 (r2 + 2 b^2) + 2 p2 a b
//   u = fx xd + cx,  v = fy yd + cy
//
// parameters = fx, fy, cx, cy, k1, k2, p1, p2 (, k3), the coefficient order of OpenCV.
// kNumRadial = 2: Kalibr radtan / COLMAP OPENCV; kNumRadial = 3: OpenCV's 5-coefficient model.
// Unprojection uses Newton's method; it is only meaningful where the distortion is monotonic.
template <int kNumRadial>
class PinholeRadTan {
  static_assert(kNumRadial == 2 || kNumRadial == 3, "2 or 3 radial coefficients");

 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int kNumDistortionParams = kNumRadial + 2;
  static constexpr int kNumParams = 4 + kNumDistortionParams;
  static constexpr const char* kName = kNumRadial == 2 ? "PinholeRadTan" : "PinholeRadTanK3";
  static constexpr const char* kDescription =
      kNumRadial == 2 ? "fx, fy, cx, cy, k1, k2, p1, p2" : "fx, fy, cx, cy, k1, k2, p1, p2, k3";
  static constexpr int kFocalXIdx = 0;
  static constexpr int kFocalYIdx = 1;
  static constexpr int kPrincipalPointColIdx = 2;
  static constexpr int kPrincipalPointRowIdx = 3;
  static constexpr bool kIsFisheye = false;
  static constexpr bool kHasAnalyticalProjection = true;

  template <
      class D,
      class DP,
      class DJ1 = Eigen::Matrix<typename D::Scalar, 2, 3>,
      class DJ2 = Eigen::Matrix<typename D::Scalar, 2, kNumParams>>
  static Eigen::Matrix<typename D::Scalar, 2, 1> project(
      const Eigen::MatrixBase<D>& pointOptical,
      const Eigen::MatrixBase<DP>& params,
      Eigen::MatrixBase<DJ1>* d_point = nullptr,
      Eigen::MatrixBase<DJ2>* d_params = nullptr) {
    validateProjectInput<D, DP, kNumParams>();
    using T = typename D::Scalar;

    SOPHUS_ENSURE(pointOptical.z() != T(0), "z(%) must not be zero.", pointOptical.z());

    const T fx = params[0];
    const T fy = params[1];
    const T zInv = T(1) / pointOptical(2);
    const T a = pointOptical(0) * zInv;
    const T b = pointOptical(1) * zInv;

    Eigen::Matrix<T, 2, 2> J_dist;
    const Eigen::Matrix<T, 2, 1> d =
        distort(a, b, params, (d_point != nullptr) ? &J_dist : nullptr);

    if (d_point) {
      // d(a, b) / d(point)
      Eigen::Matrix<T, 2, 3> J_ab;
      J_ab << zInv, T(0), -a * zInv, //
          T(0), zInv, -b * zInv;
      const Eigen::Matrix<T, 2, 2> J_uv_d = Eigen::Matrix<T, 2, 1>(fx, fy).asDiagonal();
      (*d_point) = J_uv_d * J_dist * J_ab;
    }

    if (d_params) {
      const T r2 = a * a + b * b;
      const T r4 = r2 * r2;
      d_params->setZero();
      (*d_params)(0, 0) = d(0);
      (*d_params)(1, 1) = d(1);
      (*d_params)(0, 2) = T(1);
      (*d_params)(1, 3) = T(1);
      (*d_params)(0, 4) = fx * a * r2; // k1
      (*d_params)(1, 4) = fy * b * r2;
      (*d_params)(0, 5) = fx * a * r4; // k2
      (*d_params)(1, 5) = fy * b * r4;
      (*d_params)(0, 6) = fx * T(2) * a * b; // p1
      (*d_params)(1, 6) = fy * (r2 + T(2) * b * b);
      (*d_params)(0, 7) = fx * (r2 + T(2) * a * a); // p2
      (*d_params)(1, 7) = fy * T(2) * a * b;
      if constexpr (kNumRadial == 3) {
        (*d_params)(0, 8) = fx * a * r4 * r2; // k3
        (*d_params)(1, 8) = fy * b * r4 * r2;
      }
    }

    return Eigen::Matrix<T, 2, 1>(fx * d(0) + params[2], fy * d(1) + params[3]);
  }

  template <typename D, typename DP>
  static Eigen::Matrix<typename D::Scalar, 3, 1> unproject(
      const Eigen::MatrixBase<D>& uvPixel,
      const Eigen::MatrixBase<DP>& params) {
    validateUnprojectInput<D, DP, kNumParams>();
    using T = typename D::Scalar;

    const Eigen::Matrix<T, 2, 1> target(
        (uvPixel(0) - params[2]) / params[0], (uvPixel(1) - params[3]) / params[1]);

    // Newton's method on the 2d distortion, starting from the distorted point
    Eigen::Matrix<T, 2, 1> ab = target;
    for (int i = 0; i < CameraNewtonsMethod::kMaxIterations; ++i) {
      Eigen::Matrix<T, 2, 2> J;
      const Eigen::Matrix<T, 2, 1> d = distort(ab(0), ab(1), params, &J);
      const Eigen::Matrix<T, 2, 1> step = J.inverse() * (d - target);
      ab -= step;
      if (CameraNewtonsMethod::hasConverged(step)) {
        break;
      }
    }
    return Eigen::Matrix<T, 3, 1>(ab(0), ab(1), T(1));
  }

  template <class Scalar, class DP>
  static void scaleParams(Scalar s, Eigen::DenseBase<DP>& params) {
    validateScaleParamInput<Scalar, DP, kNumParams>();
    using T = typename DP::Scalar;
    const T& scale = s;
    params[0] *= scale;
    params[1] *= scale;
    params[2] = scale * (params[2] + 0.5) - T(0.5);
    params[3] = scale * (params[3] + 0.5) - T(0.5);
    // distortion acts on normalized coordinates, it is not affected by scale
  }

  template <typename DP>
  static void
  subtractFromOrigin(typename DP::Scalar u, typename DP::Scalar v, Eigen::DenseBase<DP>& params) {
    params[2] -= u;
    params[3] -= v;
  }

 private:
  // distorted normalized coordinates, and their Jacobian wrt (a, b)
  template <typename T, class DP>
  static Eigen::Matrix<T, 2, 1> distort(
      const T& a,
      const T& b,
      const Eigen::MatrixBase<DP>& params,
      Eigen::Matrix<T, 2, 2>* J) {
    const T k1 = params[4];
    const T k2 = params[5];
    const T p1 = params[6];
    const T p2 = params[7];
    const T k3 = kNumRadial == 3 ? T(params[kNumParams - 1]) : T(0);

    const T r2 = a * a + b * b;
    const T radial = T(1) + r2 * (k1 + r2 * (k2 + r2 * k3));
    const Eigen::Matrix<T, 2, 1> d(
        a * radial + T(2) * p1 * a * b + p2 * (r2 + T(2) * a * a),
        b * radial + p1 * (r2 + T(2) * b * b) + T(2) * p2 * a * b);

    if (J) {
      const T dRadial_dr2 = k1 + r2 * (T(2) * k2 + T(3) * k3 * r2);
      const T cross = T(2) * a * b * dRadial_dr2 + T(2) * p1 * a + T(2) * p2 * b;
      (*J)(0, 0) = radial + T(2) * a * a * dRadial_dr2 + T(2) * p1 * b + T(6) * p2 * a;
      (*J)(0, 1) = cross;
      (*J)(1, 0) = cross;
      (*J)(1, 1) = radial + T(2) * b * b * dRadial_dr2 + T(6) * p1 * b + T(2) * p2 * a;
    }
    return d;
  }
};

using PinholeRadTan4 = PinholeRadTan<2>;
using PinholeRadTan5 = PinholeRadTan<3>;

} // namespace calibration
