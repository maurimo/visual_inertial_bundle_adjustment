/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <baspacho/baspacho/Solver.h>
#include <dispenso/thread_pool.h>

namespace small_thing {

// A part of the Hessian that is kept out of the factorized matrix (Optimizer::Solver_PCG_MatrixFree).
//
// The solver factorizes a matrix M whose pattern is banded: the stores of role kRoleNormal and
// kRolePreconditionerOnly (FactorStoreBase::role). The true Hessian H differs from it by a term
// that this object knows: the curvature of the stores of role kRoleMatrixFree, minus that of the
// preconditioner-only stores. Some point variables of the elimination range ("matrix-free points")
// have no off-diagonal blocks in M; their elimination is done here. The step is then
//   reduced right-hand side: BaSpaCho's partial forward substitution, plus correctReducedRhs;
//   reduced system: PCG with M_red (the factor) as preconditioner and the product
//     H_red v = L (L^T v) + addMatvec(v);
//   back-substitution: BaSpaCho's, plus backSubstitute for the matrix-free points.
// All vectors are passed as pointers that are indexed with the FULL vector offsets
// (acc.paramStart); in the reduced calls only the reduced part may be read or written.
class MatrixFreeTerm {
 public:
  virtual ~MatrixFreeTerm() = default;

  // at the current values (the linearization point), with the damping of this step: the solver
  // adds to the diagonal of M d -> d (1 + damping) + damping, and the term must use the same rule
  // for what it eliminates
  virtual void linearize(const BaSpaCho::PermutedCoalescedAccessor& acc, double damping) = 0;
  // before the forward substitution: keep the gradient of the matrix-free points, set their slots
  // to 0
  virtual void prepareRhs(double* vec) = 0;
  // after it: the reduced right-hand side minus the matrix-free points' Schur terms
  virtual void correctReducedRhs(double* vec) = 0;
  // out += (H_red - M_red) v
  virtual void addMatvec(const double* v, double* out) = 0;
  // after the back-substitution (the reduced part of vec holds the reduced solution): the steps
  // of the matrix-free points
  virtual void backSubstitute(double* vec) = 0;
};

// out += L L^T v over the lumps from span `spanIndex` on, L being the Cholesky factor held in
// `factorData` (after factor / factorFrom: L L^T is the matrix that was factorized there, damping
// included). v and out are indexed with the full vector offsets; only [spanVectorOffset(spanIndex),
// order) is used. spanIndex must start a lump.
// With a thread pool: the lumps are split into ranges of about equal data size; the L^T pass
// writes disjoint parts, the L pass accumulates into one buffer per range, summed at the end.
void addFactorProductFrom(
    const BaSpaCho::Solver& solver,
    const double* factorData,
    int64_t spanIndex,
    const double* v,
    double* out,
    dispenso::ThreadPool* threadPool = nullptr);

} // namespace small_thing
