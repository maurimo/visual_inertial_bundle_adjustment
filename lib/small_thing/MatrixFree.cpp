/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <small_thing/MatrixFree.h>

#include <Eigen/Core>
#include <dispenso/parallel_for.h>

#include <vector>

namespace small_thing {

// BaSpaCho's layout of the factor (see Solver::addMvFrom and the BLAS calls of its CPU backend):
// per lump l (columns [lumpStart[l], lumpStart[l + 1])), the diagonal block at
// chainData[chainColPtr[l]], n x n row-major, L in its lower triangle; then the blocks below the
// diagonal, one per chain from chainColPtr[l] + boardChainColOrd[boardColPtr[l] + 1], each
// (rows of its span) x n row-major at chainData[chain], rows at spanStart[chainRowSpan[chain]].
void addFactorProductFrom(
    const BaSpaCho::Solver& solver,
    const double* factorData,
    int64_t spanIndex,
    const double* v,
    double* out,
    dispenso::ThreadPool* threadPool) {
  using RowMat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using Vec = Eigen::VectorXd;
  const auto& sk = solver.skel();
  BASPACHO_CHECK_EQ(sk.spanOffsetInLump[spanIndex], 0);
  const int64_t l0 = sk.spanToLump[spanIndex];
  const int64_t numLumps = int64_t(sk.lumpStart.size()) - 1;
  const int64_t base = sk.lumpStart[l0];
  const int64_t size = sk.order() - base;
  auto belowChains = [&](int64_t l, int64_t& begin, int64_t& end) {
    const int64_t c0 = sk.chainColPtr[l];
    const int64_t b0 = sk.boardColPtr[l], b1 = sk.boardColPtr[l + 1];
    begin = c0 + sk.boardChainColOrd[b0 + 1];
    end = c0 + sk.boardChainColOrd[b1 - 1];
  };
  // y = L^T v, lump by lump (each lump writes its own segment)
  Vec y = Vec::Zero(size);
  auto ltLumps = [&](int64_t la, int64_t lb) {
    for (int64_t l = la; l < lb; l++) {
      const int64_t ls = sk.lumpStart[l], n = sk.lumpStart[l + 1] - ls;
      const Eigen::Map<const RowMat> D(factorData + sk.chainData[sk.chainColPtr[l]], n, n);
      auto yl = y.segment(ls - base, n);
      yl.noalias() += D.triangularView<Eigen::Lower>().transpose() * Eigen::Map<const Vec>(v + ls, n);
      int64_t cb, ce;
      belowChains(l, cb, ce);
      for (int64_t c = cb; c < ce; c++) {
        const int64_t span = sk.chainRowSpan[c];
        const int64_t rs = sk.spanStart[span], rows = sk.spanStart[span + 1] - rs;
        const Eigen::Map<const RowMat> Q(factorData + sk.chainData[c], rows, n);
        yl.noalias() += Q.transpose() * Eigen::Map<const Vec>(v + rs, rows);
      }
    }
  };
  // z += L y over a range of lumps (rows anywhere below: into the given buffer)
  auto lLumps = [&](int64_t la, int64_t lb, double* z) {
    for (int64_t l = la; l < lb; l++) {
      const int64_t ls = sk.lumpStart[l], n = sk.lumpStart[l + 1] - ls;
      const Eigen::Map<const RowMat> D(factorData + sk.chainData[sk.chainColPtr[l]], n, n);
      const auto yl = y.segment(ls - base, n);
      Eigen::Map<Vec>(z + ls, n).noalias() += D.triangularView<Eigen::Lower>() * yl;
      int64_t cb, ce;
      belowChains(l, cb, ce);
      for (int64_t c = cb; c < ce; c++) {
        const int64_t span = sk.chainRowSpan[c];
        const int64_t rs = sk.spanStart[span], rows = sk.spanStart[span + 1] - rs;
        const Eigen::Map<const RowMat> Q(factorData + sk.chainData[c], rows, n);
        Eigen::Map<Vec>(z + rs, rows).noalias() += Q * yl;
      }
    }
  };
  const int numThreads = threadPool ? int(threadPool->numThreads()) : 0;
  if (numThreads <= 1) {
    ltLumps(l0, numLumps);
    lLumps(l0, numLumps, out);
    return;
  }
  // ranges of lumps of about equal data size
  const int numRanges = 2 * numThreads;
  const int64_t d0 = sk.chainData[sk.chainColPtr[l0]], d1 = sk.dataSize();
  std::vector<int64_t> cut{l0};
  for (int r = 1; r < numRanges; r++) {
    const int64_t target = d0 + (d1 - d0) * r / numRanges;
    int64_t l = cut.back();
    while (l < numLumps && sk.chainData[sk.chainColPtr[l]] < target) {
      l++;
    }
    cut.push_back(l);
  }
  cut.push_back(numLumps);
  dispenso::TaskSet tasks(*threadPool);
  dispenso::parallel_for(tasks, 0, numRanges, [&](int r) { ltLumps(cut[r], cut[r + 1]); });
  std::vector<Vec> z(numRanges, Vec::Zero(size));
  dispenso::parallel_for(tasks, 0, numRanges, [&](int r) { lLumps(cut[r], cut[r + 1], z[r].data() - base); });
  Eigen::Map<Vec> o(out + base, size);
  for (const auto& zr : z) {
    o += zr;
  }
}

} // namespace small_thing
