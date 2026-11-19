/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cmath>

#include <calibration/projections/Common.h>

// Wide angle models of the unified family, with the parametrizations of Basalt
// (V. Usenko, N. Demmel, D. Cremers, "The Double Sphere Camera Model", 3DV 2018). All have the
// form
//
//   u = fx x / den + cx,  v = fy y / den + cy
//
// with a model-specific denominator `den`. Their closed form unprojection is defined only inside
// the valid image region (see each model), outside of it the returned ray has NaN components.

namespace calibration {

namespace unified_detail {

// projection from the denominator and its derivatives (ddenDp: 1x3 wrt point, ddenDq: wrt the
// model parameters after fx, fy, cx, cy)
template <class T, int kNumModelParams, class D, class DP, class DJ1, class DJ2>
Eigen::Matrix<T, 2, 1> projectWithDen(
    const Eigen::MatrixBase<D>& p,
    const Eigen::MatrixBase<DP>& params,
    const T& den,
    const Eigen::Matrix<T, 1, 3>& ddenDp,
    const Eigen::Matrix<T, 1, kNumModelParams>& ddenDq,
    Eigen::MatrixBase<DJ1>* d_point,
    Eigen::MatrixBase<DJ2>* d_params) {
  const T fx = params[0];
  const T fy = params[1];
  const T denInv = T(1) / den;
  const T mx = p(0) * denInv;
  const T my = p(1) * denInv;

  if (d_point) {
    // d(x / den) = e_x / den - x / den^2 dden
    (*d_point)(0, 0) = fx * (denInv - mx * denInv * ddenDp(0));
    (*d_point)(0, 1) = -fx * mx * denInv * ddenDp(1);
    (*d_point)(0, 2) = -fx * mx * denInv * ddenDp(2);
    (*d_point)(1, 0) = -fy * my * denInv * ddenDp(0);
    (*d_point)(1, 1) = fy * (denInv - my * denInv * ddenDp(1));
    (*d_point)(1, 2) = -fy * my * denInv * ddenDp(2);
  }
  if (d_params) {
    d_params->setZero();
    (*d_params)(0, 0) = mx;
    (*d_params)(1, 1) = my;
    (*d_params)(0, 2) = T(1);
    (*d_params)(1, 3) = T(1);
    for (int i = 0; i < kNumModelParams; ++i) {
      (*d_params)(0, 4 + i) = -fx * mx * denInv * ddenDq(i);
      (*d_params)(1, 4 + i) = -fy * my * denInv * ddenDq(i);
    }
  }
  return Eigen::Matrix<T, 2, 1>(fx * mx + params[2], fy * my + params[3]);
}

template <class Scalar, class DP>
void scaleFocalAndCenter(Scalar s, Eigen::DenseBase<DP>& params) {
  using T = typename DP::Scalar;
  const T& scale = s;
  params[0] *= scale;
  params[1] *= scale;
  params[2] = scale * (params[2] + 0.5) - T(0.5);
  params[3] = scale * (params[3] + 0.5) - T(0.5);
}

} // namespace unified_detail

// Double Sphere model.
//
//   d1 = |p|,  zeta = xi d1 + z,  d2 = sqrt(x^2 + y^2 + zeta^2)
//   den = alpha d2 + (1 - alpha) zeta
//
// parameters = fx, fy, cx, cy, xi, alpha
// Unprojection is valid for alpha <= 0.5 or r^2 <= 1 / (2 alpha - 1), r^2 = mx^2 + my^2.
class DoubleSphereProjection {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int kNumParams = 6;
  static constexpr char kName[] = "DoubleSphere";
  static constexpr char kDescription[] = "fx, fy, cx, cy, xi, alpha";
  static constexpr int kNumDistortionParams = 2;
  static constexpr int kFocalXIdx = 0;
  static constexpr int kFocalYIdx = 1;
  static constexpr int kPrincipalPointColIdx = 2;
  static constexpr int kPrincipalPointRowIdx = 3;
  static constexpr bool kIsFisheye = true;
  static constexpr bool kHasAnalyticalProjection = true;

  template <
      class D,
      class DP,
      class DJ1 = Eigen::Matrix<typename D::Scalar, 2, 3>,
      class DJ2 = Eigen::Matrix<typename D::Scalar, 2, kNumParams>>
  static Eigen::Matrix<typename D::Scalar, 2, 1> project(
      const Eigen::MatrixBase<D>& p,
      const Eigen::MatrixBase<DP>& params,
      Eigen::MatrixBase<DJ1>* d_point = nullptr,
      Eigen::MatrixBase<DJ2>* d_params = nullptr) {
    validateProjectInput<D, DP, kNumParams>();
    using T = typename D::Scalar;
    using std::sqrt;

    const T xi = params[4];
    const T alpha = params[5];
    const T xy2 = p(0) * p(0) + p(1) * p(1);
    const T d1 = sqrt(xy2 + p(2) * p(2));
    const T zeta = xi * d1 + p(2);
    const T d2 = sqrt(xy2 + zeta * zeta);
    const T den = alpha * d2 + (T(1) - alpha) * zeta;

    Eigen::Matrix<T, 1, 3> ddenDp;
    Eigen::Matrix<T, 1, 2> ddenDq;
    if (d_point || d_params) {
      // dzeta/dp = xi p / d1 + e_z, dd2/dp = (x, y, 0) / d2 + zeta / d2 dzeta/dp
      Eigen::Matrix<T, 1, 3> dzeta = xi / d1 * p.transpose();
      dzeta(2) += T(1);
      Eigen::Matrix<T, 1, 3> dd2 = zeta / d2 * dzeta;
      dd2(0) += p(0) / d2;
      dd2(1) += p(1) / d2;
      ddenDp = alpha * dd2 + (T(1) - alpha) * dzeta;
      ddenDq(0) = alpha * zeta * d1 / d2 + (T(1) - alpha) * d1; // xi
      ddenDq(1) = d2 - zeta; // alpha
    }
    return unified_detail::projectWithDen<T, 2>(p, params, den, ddenDp, ddenDq, d_point, d_params);
  }

  template <typename D, typename DP>
  static Eigen::Matrix<typename D::Scalar, 3, 1> unproject(
      const Eigen::MatrixBase<D>& uvPixel,
      const Eigen::MatrixBase<DP>& params) {
    validateUnprojectInput<D, DP, kNumParams>();
    using T = typename D::Scalar;
    using std::sqrt;

    const T xi = params[4];
    const T alpha = params[5];
    const T mx = (uvPixel(0) - params[2]) / params[0];
    const T my = (uvPixel(1) - params[3]) / params[1];
    const T r2 = mx * mx + my * my;
    const T mz = (T(1) - alpha * alpha * r2) /
        (alpha * sqrt(T(1) - (T(2) * alpha - T(1)) * r2) + T(1) - alpha);
    const T mz2 = mz * mz;
    const T k = (mz * xi + sqrt(mz2 + (T(1) - xi * xi) * r2)) / (mz2 + r2);
    return Eigen::Matrix<T, 3, 1>(k * mx, k * my, k * mz - xi);
  }

  template <class Scalar, class DP>
  static void scaleParams(Scalar s, Eigen::DenseBase<DP>& params) {
    validateScaleParamInput<Scalar, DP, kNumParams>();
    unified_detail::scaleFocalAndCenter(s, params);
  }

  template <typename DP>
  static void
  subtractFromOrigin(typename DP::Scalar u, typename DP::Scalar v, Eigen::DenseBase<DP>& params) {
    params[2] -= u;
    params[3] -= v;
  }
};

// Extended Unified Camera Model (B. Khomutenko et al., 2016).
//
//   d = sqrt(beta (x^2 + y^2) + z^2),  den = alpha d + (1 - alpha) z
//
// parameters = fx, fy, cx, cy, alpha, beta
// Unprojection is valid for alpha <= 0.5 or r^2 <= 1 / (beta (2 alpha - 1)).
class ExtendedUnifiedProjection {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int kNumParams = 6;
  static constexpr char kName[] = "ExtendedUnified";
  static constexpr char kDescription[] = "fx, fy, cx, cy, alpha, beta";
  static constexpr int kNumDistortionParams = 2;
  static constexpr int kFocalXIdx = 0;
  static constexpr int kFocalYIdx = 1;
  static constexpr int kPrincipalPointColIdx = 2;
  static constexpr int kPrincipalPointRowIdx = 3;
  static constexpr bool kIsFisheye = true;
  static constexpr bool kHasAnalyticalProjection = true;

  template <
      class D,
      class DP,
      class DJ1 = Eigen::Matrix<typename D::Scalar, 2, 3>,
      class DJ2 = Eigen::Matrix<typename D::Scalar, 2, kNumParams>>
  static Eigen::Matrix<typename D::Scalar, 2, 1> project(
      const Eigen::MatrixBase<D>& p,
      const Eigen::MatrixBase<DP>& params,
      Eigen::MatrixBase<DJ1>* d_point = nullptr,
      Eigen::MatrixBase<DJ2>* d_params = nullptr) {
    validateProjectInput<D, DP, kNumParams>();
    using T = typename D::Scalar;
    using std::sqrt;

    const T alpha = params[4];
    const T beta = params[5];
    const T xy2 = p(0) * p(0) + p(1) * p(1);
    const T d = sqrt(beta * xy2 + p(2) * p(2));
    const T den = alpha * d + (T(1) - alpha) * p(2);

    Eigen::Matrix<T, 1, 3> ddenDp;
    Eigen::Matrix<T, 1, 2> ddenDq;
    if (d_point || d_params) {
      ddenDp << alpha * beta * p(0) / d, alpha * beta * p(1) / d,
          alpha * p(2) / d + (T(1) - alpha);
      ddenDq(0) = d - p(2); // alpha
      ddenDq(1) = alpha * xy2 / (T(2) * d); // beta
    }
    return unified_detail::projectWithDen<T, 2>(p, params, den, ddenDp, ddenDq, d_point, d_params);
  }

  template <typename D, typename DP>
  static Eigen::Matrix<typename D::Scalar, 3, 1> unproject(
      const Eigen::MatrixBase<D>& uvPixel,
      const Eigen::MatrixBase<DP>& params) {
    validateUnprojectInput<D, DP, kNumParams>();
    using T = typename D::Scalar;
    using std::sqrt;

    const T alpha = params[4];
    const T beta = params[5];
    const T mx = (uvPixel(0) - params[2]) / params[0];
    const T my = (uvPixel(1) - params[3]) / params[1];
    const T r2 = mx * mx + my * my;
    const T mz = (T(1) - beta * alpha * alpha * r2) /
        (alpha * sqrt(T(1) - (T(2) * alpha - T(1)) * beta * r2) + (T(1) - alpha));
    return Eigen::Matrix<T, 3, 1>(mx, my, mz);
  }

  template <class Scalar, class DP>
  static void scaleParams(Scalar s, Eigen::DenseBase<DP>& params) {
    validateScaleParamInput<Scalar, DP, kNumParams>();
    unified_detail::scaleFocalAndCenter(s, params);
  }

  template <typename DP>
  static void
  subtractFromOrigin(typename DP::Scalar u, typename DP::Scalar v, Eigen::DenseBase<DP>& params) {
    params[2] -= u;
    params[3] -= v;
  }
};

// Unified Camera Model (C. Geyer, K. Daniilidis 2000; C. Mei, P. Rives 2007), parametrized as in
// Basalt: den = alpha |p| + (1 - alpha) z. It is Mei's model with xi = alpha / (1 - alpha) and
// gamma = f / (1 - alpha) (Kalibr "omni" camera without distortion).
//
// parameters = fx, fy, cx, cy, alpha
// Unprojection is valid for alpha <= 0.5 or r^2 <= 1 / (2 alpha - 1), r^2 = mx^2 + my^2.
class UnifiedProjection {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  static constexpr int kNumParams = 5;
  static constexpr char kName[] = "Unified";
  static constexpr char kDescription[] = "fx, fy, cx, cy, alpha";
  static constexpr int kNumDistortionParams = 1;
  static constexpr int kFocalXIdx = 0;
  static constexpr int kFocalYIdx = 1;
  static constexpr int kPrincipalPointColIdx = 2;
  static constexpr int kPrincipalPointRowIdx = 3;
  static constexpr bool kIsFisheye = true;
  static constexpr bool kHasAnalyticalProjection = true;

  template <
      class D,
      class DP,
      class DJ1 = Eigen::Matrix<typename D::Scalar, 2, 3>,
      class DJ2 = Eigen::Matrix<typename D::Scalar, 2, kNumParams>>
  static Eigen::Matrix<typename D::Scalar, 2, 1> project(
      const Eigen::MatrixBase<D>& p,
      const Eigen::MatrixBase<DP>& params,
      Eigen::MatrixBase<DJ1>* d_point = nullptr,
      Eigen::MatrixBase<DJ2>* d_params = nullptr) {
    validateProjectInput<D, DP, kNumParams>();
    using T = typename D::Scalar;
    using std::sqrt;

    const T alpha = params[4];
    const T d = sqrt(p(0) * p(0) + p(1) * p(1) + p(2) * p(2));
    const T den = alpha * d + (T(1) - alpha) * p(2);

    Eigen::Matrix<T, 1, 3> ddenDp;
    Eigen::Matrix<T, 1, 1> ddenDq;
    if (d_point || d_params) {
      ddenDp = alpha / d * p.transpose();
      ddenDp(2) += T(1) - alpha;
      ddenDq(0) = d - p(2); // alpha
    }
    return unified_detail::projectWithDen<T, 1>(p, params, den, ddenDp, ddenDq, d_point, d_params);
  }

  template <typename D, typename DP>
  static Eigen::Matrix<typename D::Scalar, 3, 1> unproject(
      const Eigen::MatrixBase<D>& uvPixel,
      const Eigen::MatrixBase<DP>& params) {
    validateUnprojectInput<D, DP, kNumParams>();
    using T = typename D::Scalar;
    using std::sqrt;

    // in Mei's form: (X, Y) = (x, y) / (z + xi |p|), with X = (1 - alpha) mx
    const T alpha = params[4];
    const T xi = alpha / (T(1) - alpha);
    const T X = (T(1) - alpha) * (uvPixel(0) - params[2]) / params[0];
    const T Y = (T(1) - alpha) * (uvPixel(1) - params[3]) / params[1];
    const T r2 = X * X + Y * Y;
    const T k = (xi + sqrt(T(1) + (T(1) - xi * xi) * r2)) / (T(1) + r2);
    return Eigen::Matrix<T, 3, 1>(k * X, k * Y, k - xi);
  }

  template <class Scalar, class DP>
  static void scaleParams(Scalar s, Eigen::DenseBase<DP>& params) {
    validateScaleParamInput<Scalar, DP, kNumParams>();
    unified_detail::scaleFocalAndCenter(s, params);
  }

  template <typename DP>
  static void
  subtractFromOrigin(typename DP::Scalar u, typename DP::Scalar v, Eigen::DenseBase<DP>& params) {
    params[2] -= u;
    params[3] -= v;
  }
};

} // namespace calibration
