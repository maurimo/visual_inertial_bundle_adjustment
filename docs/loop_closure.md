# Loop closure: PCG with the banded open-loop factor as preconditioner

Runs if and only if loop data is given: `--lc-tracks <file>` and/or `--lc-matches <file>`. `--lc-mode ba` (default;
`full` is the same; `pg` = pose graph only, a diagnostic). Code: `viba/loop_closure/` (pose graph),
`viba/problem/LoopBundle.{h,cpp}`, `viba/problem/LoopProblem.cpp` (loop-closed bundle adjustment),
`lib/small_thing/MatrixFree.{h,cpp}` and `Optimizer::Solver_PCG_MatrixFree` (solver). Tests:
`viba/loop_closure/tests/TestLoopClosure.cpp`, `viba/problem/tests/TestLoopBundle.cpp`.
Preparing the inputs: [README, Aria recordings](../README.md#aria-recordings-lamaria-slam-dunk-results),
`tools/loop_closure/lc_split_viba_input.py`.

## The constraint

VI-BA's direct solver works because the reduced system is a band in time
([elimination_order.md](elimination_order.md)). A landmark seen again minutes later joins rigs minutes apart: put
into the factorized matrix it would fill the band between them. A one-shot pose graph followed by open-loop BA does
not work either: the converged open-loop optimum carries the heading drift (points out of view are never found
again), so the open-loop BA pulls the corrected trajectory back. **The loop terms must stay in the cost to the end,
without entering the factorization.**

Alternatives considered and rejected: ADMM in consensus or sharing form (it reaches the joint optimum only in the
limit, at a rate set by the ratio of loop stiffness to the open-loop soft-mode stiffness, about 1e4-1e6 here:
hundreds of outer iterations); block-coordinate descent between BA and a pose graph (the same slow tail along the
drift mode); low-rank updates of the Cholesky factor (an off-band update fills the factor along the elimination
tree); loop variables ordered last (a dense border of gigabytes for thousands of loop dimensions).

## Pipeline

After the open-loop optimization of the last pass (scale-knot stage and main call, drift knots folded):

1. **Loop key rigs and candidates** (`LoopClosure.cpp`). A grid of key rigs every `--lc-lckr-sec` (20 s) among
   those with matches; for each, earlier rigs more than `--lc-min-gap` (30 s) earlier, within `--lc-radius` (25 m)
   on the current trajectory, up to `--lc-partners` (2) verified partners.
2. **Verification**: per camera pair, bearings from pixels, 5-point RANSAC, refinement, a gate at 20 deg from the
   open-loop relative rotation; at least `--lc-min-inliers` (20). **Scale-free**: one loop edge per rig pair, and
   the length of the 5-point translation is never used (only its direction), so the stereo baseline gives no scale.
3. **Pose graph, stage 1**: odometry edges between consecutive nodes (open-loop relative 4-DoF motion, uncertainty
   from `OdometryNoise.h`: a random walk in time and distance, scaled by the tracking quality of the interval) and
   loop edges (relative yaw + direction of the other camera centre).
4. **Pose graph, stage 2**: started from stage 1; odometry edges plus one point per inlier match of the accepted
   pairs (inverse depth in the earlier rig), with reprojection errors in both rigs.
5. **Robustness** (both stages): GNC with Geman-McClure, one weight per loop pair, then the exact form (pairs below
   the threshold at weight 1, the others off). A pair turned by 6 deg is switched off in the tests.
6. **Two variants** (`--lc-variant poses|knots`): the 4-DoF node poses as variables (yaw correction on top of the
   open-loop attitude), or only drift knots, with the key rigs held at the open-loop trajectory and the node states
   functions of the knots. The correction is mapped back to every key rig **through the drift-knot transform**
   ([drift_knots.md](drift_knots.md)): it turns each step by the yaw accumulated up to it, so attitude and
   direction of motion turn together (plain interpolation of the corrections would turn attitudes without turning
   the steps, and every inertial factor would object).
7. **Loop-closed bundle adjustment** (below), `--lc-ba-iters` (50) LM iterations with the drift knots active.

With `--lc-tracks` the pose graph's correspondences are derived from the tracks (pairs between consecutive visits of
a loop point, at most `--lc-track-pairs` 20 per pair of visits), and the loop points of the bundle adjustment come
from the tracks.

## The loop-closed bundle adjustment

**Variables.** One point variable per loop landmark, with all its observations from all its visits (in inverse depth
in the rig of its first observation). Its observations live in one factor store of role `kRoleMatrixFree`: cost and
gradient as any factor, but their curvature is never written into BaSpaCho's matrix and they register no blocks, so
the factorized pattern stays banded.

**Virtual points.** For each loop point and each visit seen from at least 2 rigs, a "virtual point" of the
elimination range, hosted in a rig of the visit, carries preconditioner-only copies of that visit's observations
(role `kRolePreconditionerOnly`: Hessian only, no cost, no gradient). The factorized matrix is then

$$ M = \text{banded open-loop model} + \textstyle\sum_g S_g $$

and the true reduced Hessian is $H = M + \sum_L S_L - \sum_g S_g$, where $S_L$ is the Schur complement of loop
point $L$ with all its observations and $S_g$ that of the virtual point of visit $g$. Each $S_L - \sum_{g \in L}
S_g$ is positive semi-definite of rank at most $3(k - 1)$ for $k$ visits: the preconditioned operator is the
identity plus a low-rank positive term per loop point, so PCG needs about as many iterations as there are
independent loop-stiff directions. In the synthetic test, virtual points take PCG from 176 to 113 iterations.
Virtual points are re-set from their loop point before every iteration.

**The step** (`Solver_PCG_MatrixFree`):

1. BaSpaCho factorizes $M$ completely, in place (points, then the reduced system): no copy of the reduced matrix.
2. Right-hand side: the loop points' gradient is taken out, BaSpaCho's partial forward substitution, then minus
   $\sum_L V_L (H^\lambda_{LL})^{-1} g_L$.
3. PCG on the reduced system: preconditioner = the triangular solves with the factor; product
   $$ H_{\text{red}}\,v = L(L^\top v) + \textstyle\sum_L S_L v - \sum_g S_g v $$
   BaSpaCho has no triangular multiply, so `small_thing::addFactorProductFrom` computes $L(L^\top v)$ directly on
   the factor's block layout (parallel over ranges of lumps). $LL^\top$ is exactly the damped $M$ that was
   factorized; $S_g$ is subtracted with the virtual factors' own linearization, damped as the solver damps them.
4. Back-substitution: BaSpaCho's for the ordinary points, $\Delta p_L = (H^\lambda_{LL})^{-1}(g_L - V_L^\top \Delta x)$
   for the loop points.

The PCG tolerance is adaptive (Eisenstat-Walker, choice 2: $\eta_k = 0.9\,(|g_k| / |g_{k-1}|)^2$ within
[`--lc-pcg-eta-min` 1e-6, `--lc-pcg-eta-max` 0.1], `--lc-pcg-iters` 100 the cap): in the synthetic LM test 211 PCG
iterations in all against 871 for a fixed 1e-8, same final cost. Damping: LM's rule on $M$'s diagonal and on each
loop point's block. Tested: one PCG step equals the dense solve of the true damped system to 3e-12.

**Drift knots across many intervals.** A loop observation joins its host and its rig through every knot interval
between them. `LoopPathCache` computes once per knot state the true path of all key rigs and, per knot, the sums
needed for exact Jacobians of the relative transform over any number of intervals, so the knots stay a neutral
over-parametrization for the loop factors too (gauge test: cost 3402.13 -> 3402.13).

**Robustness.** Each loop observation has VI-BA's reprojection loss times a weight per loop pair. For the first
`--lc-gnc-iters` (5) iterations the weights follow GNC-GM (set between iterations only, so the cost comparison inside
an iteration stays consistent; the Huber cut-off is widened by $\sqrt\mu$ meanwhile); then the exact form.

## Results (LaMAria sequence_1_20, 21 min)

| trajectory | score % (ATE m) |
|---|---|
| SLAM Dunk VIO before its pose graph | 34.1 (3.717) |
| SLAM Dunk pose graph | 33.8 (4.007) |
| SLAM Dunk loop-closed batch | 91.1 (0.242) |
| VI-BA on SLAM Dunk's observations, from its pose graph: open loop / **loop-closed** | 89.4 (0.283) / **98.3 (0.166)** |
| VI-BA on SLAM Dunk's observations, from its VIO before the pose graph: open loop / **loop-closed** | 90.0 (0.262) / **97.3 (0.165)** |
| VI-BA on a filter's observations: open loop / loop-closed with SIFT loops / with LoMa loops | 71.9 / 75.8 / 74.1 |

On SLAM Dunk's measurements (its tracking points and its revisit points, split at gaps of more than 27 s into 2095
loop points with 26 217 observations), both starts, 4 m apart, reach the same solution and score 6-7 points above
SLAM Dunk's own batch, in 6-7 min and 1 GB. With the same optimizer, a filter's observations reach 72 % open loop and
76 % loop-closed: on this recording the gap is in the input measurements, not in the loop closing.

## Open

- Our pose-graph stage found only 1 verified pair from the consecutive-visit correspondences of SLAM Dunk's data
  (single key-frame observations rarely make a 5-point set); the loop BA did the correction alone.
- The loop observations use the global-shutter visual factor (no row-time model), and the observing rig's velocity
  is not one of their variables.
- The odometry uncertainty of the pose graph is not fitted (placeholder values).
- `CameraModelParam::unprojectNoChecks` is inaccurate near the image border; the 5-point stage refines bearings
  against `project()`, other users of it were not checked.
- The product $L(L^\top v)$ reads the whole reduced factor twice per PCG iteration.
