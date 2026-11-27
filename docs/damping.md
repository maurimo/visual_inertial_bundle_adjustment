# Damping: the Levenberg-Marquardt damping as a hidden prior

VI-BA optimizes with Levenberg-Marquardt (`lib/small_thing/Optimizer.cpp`). This note explains what the damping does
to a problem of this size, why it held whole classes of corrections back, and what was changed about it.

## The damping of small_thing

At every iteration the diagonal of the Gauss-Newton Hessian is changed as (`Optimizer::addDamping`)

$$ H_{ii} \leftarrow H_{ii}\,(1 + \lambda) + \lambda $$

so the step pays $\lambda\,(H_{ii} + 1)\,\Delta x_i^2$ for moving variable $i$. There are two parts:

- a **relative** part $\lambda H_{ii}$, proportional to the curvature the data gives the variable (as in
  Marquardt's scaling);
- an **absolute** part $\lambda$, the same for every variable whatever its curvature.

Either way the damping is a **temporary prior centred on the current values of all the variables**, which LM lowers
as the steps succeed. It decides *when* a correction is allowed to happen, and that is where the two problems below
come from.

## Problem 1: the absolute part freezes far points, and far points freeze the heading

With points stored as world coordinates, low-parallax tracks keep being pushed farther away (their robust cost keeps
falling towards infinity). On LaMAria's R_12_10cp, 270-590 of about 97 000 points ended beyond 100 km and 110-130
beyond $10^9$ m. To turn the cameras that see such a point by 0.5 deg, the point has to move sideways by 0.0087
times its distance, i.e. by about $10^{11}$ m. Its curvature along that direction is about $10^{-22}$, far below the
absolute damping ($\lambda = 10^{-5} \dots 10^{-9}$ in the stock schedule): the point cannot move, so neither can the
heading of the cameras that see it. A few hundred such "fixed stars" pinned the heading of the whole trajectory.

What showed it: VI-BA's own cost preferred a better solution (by 0.36 %), but no change of the sensor model moved
the optimizer towards it (calibration random walks, windows, bias walks, loss, more passes: 40-45 % against 82 % for
the same tracks from another start). Following the cost along the straight path between the two solutions, one
observation at a time, put the whole rise of the visual term on about 100 points at $10^9$-$10^{14}$ m.

**Why another optimizer (SLAM Dunk's batch, Ceres) does not get stuck on the same points.** It has world points and
far points too (763 of 99 288 beyond 100 km), but Ceres' LM damping is
$D^2 = \mathrm{clamp}(\mathrm{diag}\,J^\top J, 10^{-6}, 10^{32}) / \text{radius}$, purely relative (Jacobi
scaled), and the trust radius grows by 3 per good step from $10^4$ to $10^{16}$. After about 130 iterations per pass
the damping on a far point is around $10^{-22}$ and the step is close to Gauss-Newton; VI-BA's floor of $10^{-9}$ is
never that low.

**Fix**: not a change of the damping, but of what is damped: points live in a host rig, far points as inverse depth
([host_points.md](host_points.md)). A point at infinity is then an ordinary value that turns with its host. Removing
the absolute part of the damping, or bringing world points back to 1000 km before every step, each helped
(58.7 % and 68.3 % against 40.9 %) but neither converged; the host-rig form converged in 28 iterations.

## Problem 2: slow drifts are soft modes, and the damping charges every pose for them

A slow drift of the trajectory (scale, heading, position) that starts somewhere in the recording moves every later
pose by an amount that grows with time. The data hardly sees it (a drifted trajectory with a correct calibration is
nearly free: 13 units of cost out of 111 560 for a +-0.25 % scale ramp on R_11_5cp), while the damping charges
each of the thousands of moved variables. So the correction waits until $\lambda$ is very small: in the tests, 6-8
iterations do nothing, then it crawls. With an injected heading drift of 1.5 deg, nothing moves until the damping is
at $10^{-10}$; at a damping held at $10^{-6}$ it is never corrected. Stiffer calibration random walks change
nothing: it is not the freedom of the calibration but the damping.

**Fix**: over-parametrizations that turn such a drift into a few variables, so that it costs one damped variable
instead of all the poses: scale knots ([scale_knots.md](scale_knots.md)) and drift knots
([drift_knots.md](drift_knots.md)).

## Problem 3: the stock schedule was too cautious

On R_11_5cp with the stock schedule (start $10^{-5}$, divided by 3 per good step, floor $10^{-9}$), iterations 0-6
leave the trajectory where it is, then a tail of 30 iterations buys 1 unit of cost out of 106 448 and 7 cm. Every
step gains 150-198 % of the reduction the model predicts. LM's predicted gain, $\tfrac12 g^\top \Delta$, counts the
damping as if it were curvature:

$$ \frac{\text{actual}}{\text{predicted}} = 2 - \frac{\text{damping}}{\text{curvature along the step}} $$

and the damping measured along each step was 40-50 % of the curvature at every level from $10^{-6}$ to $10^{-9}$
(above 95 % in the tail), nearly all of it the relative part on the poses. No step was ever rejected: the schedule
lowered $\lambda$ by 3 when the step had been twice as good as promised.

**Fix: the schedule and the stopping rule now in the defaults** (`viba/common/Settings.h`):

| option | default | meaning |
|---|---|---|
| `--initial-damping` | 1e-5 | damping at the start of the first pass (a low damping at the very first step from a filter's state is harmful: cost 5.4e6 -> 2.7e6 instead of 1.1e5) |
| `--damping-good-factor` | 0.1 | damping multiplied by this after a good step |
| `--min-damping` | 1e-11 | floor |
| `--later-pass-initial-damping` | 1e-9 | a later pass starts from the previous result |
| `--absolute-cost-tolerance`, `--relative-cost-tolerance` | 0.02, 1e-8 | counted only once the damping is at its floor |
| `--stop-step-mm` | 0.5 | also converged when no key rig moves by more than this |

Three tolerance hits in a row stop. R_11_5cp, two passes: 71 -> 24 iterations, 881 -> 324 s, the same solution
(within 0.1 mm of the converged one; ATE 0.140 m throughout).

## A related trap: iterative solvers at low damping

VI-BA used to switch from the direct solver to PCG with Gauss-Seidel above 20 000 key rigs. At low damping
Gauss-Seidel and Jacobi do not reduce the residual (R_11_5cp, 12 iterations: cost 107 548 / 111 975 against 106 448
for the direct solver), so a large recording would stall. The switch is gone: the solver is `direct` unless
`--linear-solver` says otherwise. The memory of the direct factor grows linearly with the key rigs as long as the
elimination order follows time ([elimination_order.md](elimination_order.md)).
