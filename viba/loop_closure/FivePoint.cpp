/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/loop_closure/FivePoint.h>

#include <Eigen/Dense>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <random>

namespace visual_inertial_ba::loop {

namespace {

// Polynomials of degree <= 3 in (x, y, z): 20 coefficients. Monomials: the ten cubics first, then
// the ten of degree <= 2 (the basis of the action matrix: x^2, xy, xz, y^2, yz, z^2, x, y, z, 1).
constexpr int kNumMonomials = 20;
constexpr int kMonomials[kNumMonomials][3] = {
    {3, 0, 0}, {2, 1, 0}, {2, 0, 1}, {1, 2, 0}, {1, 1, 1}, {1, 0, 2}, {0, 3, 0},
    {0, 2, 1}, {0, 1, 2}, {0, 0, 3}, {2, 0, 0}, {1, 1, 0}, {1, 0, 1}, {0, 2, 0},
    {0, 1, 1}, {0, 0, 2}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {0, 0, 0}};
using Poly = std::array<double, kNumMonomials>;

struct MonomialTable {
  int index[4][4][4];
  MonomialTable() {
    for (auto& a : index) {
      for (auto& b : a) {
        for (int& c : b) {
          c = -1;
        }
      }
    }
    for (int i = 0; i < kNumMonomials; i++) {
      index[kMonomials[i][0]][kMonomials[i][1]][kMonomials[i][2]] = i;
    }
  }
};
const MonomialTable& table() {
  static const MonomialTable t;
  return t;
}

Poly linear(double a, double b, double c, double d) {
  Poly p{};
  const auto& t = table();
  p[t.index[1][0][0]] = a;
  p[t.index[0][1][0]] = b;
  p[t.index[0][0][1]] = c;
  p[t.index[0][0][0]] = d;
  return p;
}

// product, which must stay of degree <= 3
Poly mul(const Poly& p, const Poly& q) {
  Poly r{};
  const auto& t = table();
  for (int i = 0; i < kNumMonomials; i++) {
    if (p[i] == 0.0) {
      continue;
    }
    for (int j = 0; j < kNumMonomials; j++) {
      if (q[j] == 0.0) {
        continue;
      }
      const int a = kMonomials[i][0] + kMonomials[j][0];
      const int b = kMonomials[i][1] + kMonomials[j][1];
      const int c = kMonomials[i][2] + kMonomials[j][2];
      if (a + b + c > 3) {
        continue; // (cannot happen for the products used here)
      }
      r[t.index[a][b][c]] += p[i] * q[j];
    }
  }
  return r;
}

Poly add(const Poly& p, const Poly& q, double s = 1.0) {
  Poly r;
  for (int i = 0; i < kNumMonomials; i++) {
    r[i] = p[i] + s * q[i];
  }
  return r;
}

Mat33 skew(const Vec3& v) {
  Mat33 m;
  m << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
  return m;
}

Mat33 expSO3(const Vec3& w) {
  const double th = w.norm();
  if (th < 1e-12) {
    return Mat33::Identity() + skew(w);
  }
  const Mat33 K = skew(w / th);
  return Mat33::Identity() + std::sin(th) * K + (1.0 - std::cos(th)) * K * K;
}

// two unit vectors orthogonal to v (and to each other)
void tangentBasis(const Vec3& v, Vec3& b1, Vec3& b2) {
  const Vec3 a = std::abs(v.x()) < 0.9 ? Vec3::UnitX() : Vec3::UnitY();
  b1 = v.cross(a).normalized();
  b2 = v.cross(b1).normalized();
}

Mat33 essentialOf(const RelativePose& p) {
  return skew(p.t) * p.R;
}

double median(std::vector<double> v) {
  if (v.empty()) {
    return 0.0;
  }
  std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
  return v[v.size() / 2];
}

// Gauss-Newton on (rotation, direction of t) minimizing the angular epipolar errors of the pairs
double refinePose(
    RelativePose& pose,
    const std::vector<Vec3>& f1,
    const std::vector<Vec3>& f2,
    const std::vector<int>& indices,
    int iterations,
    double* sigmaRot = nullptr,
    double* sigmaDir = nullptr) {
  if (indices.size() < 6) {
    return -1.0;
  }
  auto residuals = [&](const RelativePose& p, Eigen::VectorXd& r) {
    const Mat33 E = essentialOf(p);
    r.resize(2 * indices.size());
    for (size_t k = 0; k < indices.size(); k++) {
      const Vec3& a = f1[indices[k]];
      const Vec3& b = f2[indices[k]];
      const Vec3 n = E * a, m = E.transpose() * b;
      const double e = b.dot(n);
      r[2 * k] = n.norm() > 0.0 ? e / n.norm() : 0.0;
      r[2 * k + 1] = m.norm() > 0.0 ? e / m.norm() : 0.0;
    }
  };
  auto perturbed = [](const RelativePose& p, const Eigen::Matrix<double, 5, 1>& d) {
    Vec3 b1, b2;
    tangentBasis(p.t, b1, b2);
    RelativePose q;
    q.R = expSO3(d.head<3>()) * p.R;
    q.t = (p.t + d[3] * b1 + d[4] * b2).normalized();
    return q;
  };
  Eigen::VectorXd r0, r1, r2;
  double lambda = 1e-6;
  residuals(pose, r0);
  auto jacobian = [&](Eigen::MatrixXd& J) {
    J.resize(r0.size(), 5);
    const double h = 1e-7;
    for (int c = 0; c < 5; c++) {
      Eigen::Matrix<double, 5, 1> d = Eigen::Matrix<double, 5, 1>::Zero();
      d[c] = h;
      residuals(perturbed(pose, d), r1);
      d[c] = -h;
      residuals(perturbed(pose, d), r2);
      J.col(c) = (r1 - r2) / (2.0 * h);
    }
  };
  for (int it = 0; it < iterations; it++) {
    Eigen::MatrixXd J;
    jacobian(J);
    Eigen::Matrix<double, 5, 5> H = J.transpose() * J;
    const Eigen::Matrix<double, 5, 1> g = J.transpose() * r0;
    bool improved = false;
    for (int attempt = 0; attempt < 10 && !improved; attempt++) {
      Eigen::Matrix<double, 5, 5> Hd = H;
      Hd.diagonal() *= 1.0 + lambda;
      Hd.diagonal().array() += 1e-12;
      const Eigen::Matrix<double, 5, 1> d = -Hd.ldlt().solve(g);
      const RelativePose cand = perturbed(pose, d);
      residuals(cand, r1);
      if (r1.squaredNorm() <= r0.squaredNorm()) {
        const bool tiny = r0.squaredNorm() - r1.squaredNorm() < 1e-14 * r0.squaredNorm();
        pose = cand;
        r0 = r1;
        if (tiny) {
          it = iterations; // converged
        }
        lambda = std::max(lambda * 0.1, 1e-12);
        improved = true;
      } else {
        lambda *= 10.0;
      }
    }
    if (!improved) {
      break;
    }
  }
  if (sigmaRot != nullptr && sigmaDir != nullptr) {
    Eigen::MatrixXd J;
    jacobian(J);
    const double dof = std::max<double>(double(r0.size()) - 5.0, 1.0);
    const Eigen::Matrix<double, 5, 5> H = J.transpose() * J;
    const Eigen::Matrix<double, 5, 5> C =
        (r0.squaredNorm() / dof) * H.ldlt().solve(Eigen::Matrix<double, 5, 5>::Identity());
    const Eigen::SelfAdjointEigenSolver<Mat33> er(C.topLeftCorner<3, 3>());
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> ed(C.bottomRightCorner<2, 2>());
    *sigmaRot = std::sqrt(std::max(er.eigenvalues().maxCoeff(), 0.0));
    *sigmaDir = std::sqrt(std::max(ed.eigenvalues().maxCoeff(), 0.0));
  }
  return r0.squaredNorm();
}

} // namespace

std::vector<Mat33> fivePointEssential(const Vec3* f1, const Vec3* f2) {
  // epipolar rows: f2^T E f1 with E row-major
  Eigen::Matrix<double, 5, 9> Q;
  for (int k = 0; k < 5; k++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        Q(k, 3 * i + j) = f2[k][i] * f1[k][j];
      }
    }
  }
  const Eigen::JacobiSVD<Eigen::Matrix<double, 9, 9>> svd(
      Q.transpose() * Q, Eigen::ComputeFullV);
  // the four right singular vectors of the smallest singular values
  const Eigen::Matrix<double, 9, 9>& V = svd.matrixV();
  Mat33 basis[4];
  for (int b = 0; b < 4; b++) {
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        basis[b](i, j) = V(3 * i + j, 5 + b);
      }
    }
  }
  Poly E[3][3];
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 3; j++) {
      E[i][j] = linear(basis[0](i, j), basis[1](i, j), basis[2](i, j), basis[3](i, j));
    }
  }
  Eigen::Matrix<double, 10, kNumMonomials> M;
  {
    // det E
    Poly det = mul(E[0][0], add(mul(E[1][1], E[2][2]), mul(E[1][2], E[2][1]), -1.0));
    det = add(det, mul(E[0][1], add(mul(E[1][0], E[2][2]), mul(E[1][2], E[2][0]), -1.0)), -1.0);
    det = add(det, mul(E[0][2], add(mul(E[1][0], E[2][1]), mul(E[1][1], E[2][0]), -1.0)));
    for (int c = 0; c < kNumMonomials; c++) {
      M(0, c) = det[c];
    }
    // 2 E E^T E - tr(E E^T) E
    Poly EEt[3][3];
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        Poly s{};
        for (int k = 0; k < 3; k++) {
          s = add(s, mul(E[i][k], E[j][k]));
        }
        EEt[i][j] = s;
      }
    }
    const Poly tr = add(add(EEt[0][0], EEt[1][1]), EEt[2][2]);
    for (int i = 0; i < 3; i++) {
      for (int j = 0; j < 3; j++) {
        Poly s{};
        for (int k = 0; k < 3; k++) {
          s = add(s, mul(EEt[i][k], E[k][j]), 2.0);
        }
        s = add(s, mul(tr, E[i][j]), -1.0);
        for (int c = 0; c < kNumMonomials; c++) {
          M(1 + 3 * i + j, c) = s[c];
        }
      }
    }
  }
  // cubic monomials = -B * (monomials of degree <= 2)
  const Eigen::Matrix<double, 10, 10> C = M.leftCols<10>();
  const Eigen::FullPivLU<Eigen::Matrix<double, 10, 10>> lu(C);
  if (!lu.isInvertible()) {
    return {};
  }
  const Eigen::Matrix<double, 10, 10> B = lu.solve(M.rightCols<10>());
  // action matrix of the multiplication by x on (x^2, xy, xz, y^2, yz, z^2, x, y, z, 1)
  Eigen::Matrix<double, 10, 10> A = Eigen::Matrix<double, 10, 10>::Zero();
  const auto& t = table();
  for (int r = 0; r < 10; r++) {
    const int* m = kMonomials[10 + r];
    const int idx = t.index[m[0] + 1][m[1]][m[2]];
    if (idx < 10) {
      A.row(r) = -B.row(idx);
    } else {
      A(r, idx - 10) = 1.0;
    }
  }
  const Eigen::EigenSolver<Eigen::Matrix<double, 10, 10>> es(A);
  std::vector<Mat33> out;
  for (int k = 0; k < 10; k++) {
    const std::complex<double> ev = es.eigenvalues()[k];
    if (std::abs(ev.imag()) > 1e-8 * std::max(1.0, std::abs(ev.real()))) {
      continue;
    }
    const Eigen::Matrix<double, 10, 1> b = es.eigenvectors().col(k).real();
    if (std::abs(b[9]) < 1e-14) {
      continue;
    }
    const double x = b[6] / b[9], y = b[7] / b[9], z = b[8] / b[9];
    Mat33 Es = x * basis[0] + y * basis[1] + z * basis[2] + basis[3];
    const double n = Es.norm();
    if (n > 0.0 && std::isfinite(n)) {
      out.push_back(Es / n);
    }
  }
  return out;
}

double epipolarAngularError(const Mat33& E, const Vec3& f1, const Vec3& f2) {
  const Vec3 n = E * f1, m = E.transpose() * f2;
  const double e = std::abs(f2.dot(n));
  const double nn = n.norm(), mn = m.norm();
  if (!(nn > 0.0) || !(mn > 0.0)) {
    return std::numeric_limits<double>::infinity();
  }
  return 0.5 * (e / nn + e / mn);
}

void triangulateDepths(
    const RelativePose& pose,
    const Vec3& f1,
    const Vec3& f2,
    double& d1,
    double& d2) {
  // d2 f2 = R (d1 f1) + t: least squares of d1 a + d2 b = -t, a = R f1, b = -f2
  const Vec3 a = pose.R * f1, b = -f2;
  const double aa = a.dot(a), ab = a.dot(b), bb = b.dot(b);
  const double at = a.dot(pose.t), bt = b.dot(pose.t);
  const double det = aa * bb - ab * ab;
  if (std::abs(det) < 1e-12) {
    d1 = d2 = std::numeric_limits<double>::infinity();
    return;
  }
  d1 = (-at * bb + bt * ab) / det;
  d2 = (-bt * aa + at * ab) / det;
}

int decomposeEssential(
    const Mat33& E,
    const std::vector<Vec3>& f1,
    const std::vector<Vec3>& f2,
    const std::vector<int>& indices,
    RelativePose& pose,
    const Mat33* rotationPrior) {
  const Eigen::JacobiSVD<Mat33> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Mat33 U = svd.matrixU(), V = svd.matrixV();
  if (U.determinant() < 0.0) {
    U.col(2) *= -1.0;
  }
  if (V.determinant() < 0.0) {
    V.col(2) *= -1.0;
  }
  Mat33 W;
  W << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
  Mat33 Rs[2] = {U * W * V.transpose(), U * W.transpose() * V.transpose()};
  int numRs = 2;
  if (rotationPrior != nullptr) {
    auto angle = [&](const Mat33& R) {
      return std::acos(std::clamp(0.5 * ((R * rotationPrior->transpose()).trace() - 1.0), -1.0, 1.0));
    };
    if (angle(Rs[1]) < angle(Rs[0])) {
      Rs[0] = Rs[1];
    }
    numRs = 1;
  }
  const Vec3 t = U.col(2);
  int best = 0;
  for (int ri = 0; ri < numRs; ri++) {
    const Mat33& R = Rs[ri];
    for (const double s : {1.0, -1.0}) {
      RelativePose cand;
      cand.R = R;
      cand.t = s * t;
      int front = 0;
      for (const int i : indices) {
        double d1, d2;
        triangulateDepths(cand, f1[i], f2[i], d1, d2);
        front += (d1 > 0.0 && d2 > 0.0) ? 1 : 0;
      }
      if (front > best) {
        best = front;
        pose = cand;
      }
    }
  }
  return best;
}

RansacResult ransacRelativePose(
    const std::vector<Vec3>& f1,
    const std::vector<Vec3>& f2,
    const RansacSettings& settings) {
  RansacResult result;
  const int n = int(f1.size());
  if (n < 5 || f2.size() != f1.size()) {
    return result;
  }
  std::mt19937 gen(settings.seed);
  std::uniform_int_distribution<int> pick(0, n - 1);
  int bestCount = 0;
  double bestScore = std::numeric_limits<double>::infinity();
  Mat33 bestE = Mat33::Zero();
  int maxIterations = settings.maxIterations;
  for (int it = 0; it < maxIterations; it++) {
    int sample[5];
    for (int k = 0; k < 5; k++) {
      bool dup;
      do {
        sample[k] = pick(gen);
        dup = false;
        for (int l = 0; l < k; l++) {
          dup |= sample[l] == sample[k];
        }
      } while (dup);
    }
    Vec3 a[5], b[5];
    for (int k = 0; k < 5; k++) {
      a[k] = f1[sample[k]];
      b[k] = f2[sample[k]];
    }
    for (const Mat33& E : fivePointEssential(a, b)) {
      int count = 0;
      double score = 0.0; // MSAC
      const double thr = settings.thresholdRad;
      for (int i = 0; i < n; i++) {
        const double e = epipolarAngularError(E, f1[i], f2[i]);
        if (e < thr) {
          count++;
          score += e * e;
        } else {
          score += thr * thr;
        }
      }
      if (score < bestScore) {
        bestScore = score;
        bestCount = count;
        bestE = E;
        // adaptive number of iterations
        const double w = double(count) / n;
        const double pAll = std::pow(w, 5);
        if (pAll > 1e-12 && pAll < 1.0) {
          const double needed = std::log(1.0 - settings.confidence) / std::log(1.0 - pAll);
          maxIterations = std::min(maxIterations, std::max(it + 1, int(std::ceil(needed))));
        }
      }
    }
  }
  if (bestCount < 5) {
    return result;
  }
  auto inliersOf = [&](const Mat33& E) {
    std::vector<int> in;
    for (int i = 0; i < n; i++) {
      if (epipolarAngularError(E, f1[i], f2[i]) < settings.thresholdRad) {
        in.push_back(i);
      }
    }
    return in;
  };
  std::vector<int> inliers = inliersOf(bestE);
  RelativePose pose;
  const Mat33* prior = settings.useRotationPrior ? &settings.rotationPrior : nullptr;
  if (decomposeEssential(bestE, f1, f2, inliers, pose, prior) == 0) {
    return result;
  }
  for (int round = 0; round < 2; round++) {
    refinePose(pose, f1, f2, inliers, settings.refineIterations);
    inliers = inliersOf(essentialOf(pose));
  }
  if (prior != nullptr && settings.useTranslationPrior) {
    // a second start from the prior, on the same inliers; the lower cost wins
    RelativePose fromPrior;
    fromPrior.R = *prior;
    fromPrior.t = settings.translationPrior.normalized();
    RelativePose current = pose;
    const double costCurrent = refinePose(current, f1, f2, inliers, settings.refineIterations);
    const double costPrior = refinePose(fromPrior, f1, f2, inliers, settings.refineIterations);
    pose = costPrior >= 0.0 && costPrior < costCurrent ? fromPrior : current;
    inliers = inliersOf(essentialOf(pose));
  }
  refinePose(
      pose,
      f1,
      f2,
      inliers,
      settings.refineIterations,
      &result.sigmaRotationRad,
      &result.sigmaDirectionRad);
  // cheirality: drop the inliers clearly behind a camera (those with almost no parallax cannot
  // tell, and are kept: they constrain the rotation)
  std::vector<int> kept;
  std::vector<double> parallax;
  for (const int i : inliers) {
    const double angle = std::acos(std::clamp((pose.R * f1[i]).dot(f2[i]), -1.0, 1.0));
    double d1, d2;
    triangulateDepths(pose, f1[i], f2[i], d1, d2);
    if ((d1 < 0.0 || d2 < 0.0) && angle > 4.0 * settings.thresholdRad) {
      continue;
    }
    kept.push_back(i);
    parallax.push_back(angle);
  }
  const Mat33 E = essentialOf(pose);
  double sum2 = 0.0;
  for (const int i : kept) {
    const double e = epipolarAngularError(E, f1[i], f2[i]);
    sum2 += e * e;
  }
  result.ok = kept.size() >= 5;
  if (prior != nullptr &&
      std::acos(std::clamp(0.5 * ((pose.R * prior->transpose()).trace() - 1.0), -1.0, 1.0)) >
          settings.maxPriorAngleRad) {
    result.ok = false;
  }
  result.pose = pose;
  result.inliers = kept;
  result.medianParallaxRad = median(parallax);
  result.rmsErrorRad = kept.empty() ? 0.0 : std::sqrt(sum2 / kept.size());
  return result;
}

} // namespace visual_inertial_ba::loop
