# Drift knots (4 DoF): an exact, neutral over-parametrization of slow drifts

`--yaw-knot-sec` (default 30; 0: off), `--yaw-knot-shift-std` (1 m/s; 0: yaw only), `--yaw-knot-rate-std`
(1e-2 rad/s), `--yaw-knot-rate-rw-std` (0). Code: `viba/problem/YawDrift.h`,
`SingleSessionProblem::yawKnots_*`. Open loop only (see [loop_closure.md](loop_closure.md) for how the loop-closed
problem keeps them).

## The idea

The slowest modes of an open-loop visual-inertial trajectory are a drift of the heading (about the vertical) and of
the position, starting anywhere in the recording. Correcting one moves every later pose, which the damping charges
pose by pose ([damping.md](damping.md)). Instead, give the drift its own variables: a knot holds, on its time
interval, a **yaw rate** $w_k$ [rad/s] and a **position drift velocity** $u_k$ [m/s] (4 numbers). The integral of
the rate from the first rig, $\Psi(t)$, is exact and additive. The *true* states are a transform of the variables
that depends only on time and on the knots:

$$
R^{\text{true}}_i = R_z(\Psi_i)\,R_i,\qquad v^{\text{true}}_i = R_z(\Psi_i)\,v_i,\qquad
p^{\text{true}}_i = q_i + R_z(\Psi_i)\,(p_i - \text{ref}_i)
$$

where $\text{ref}$ is a frozen copy of the positions (taken with the knots at 0) and $q$ is that copy with the drift
applied progressively from the start,

$$ q_0 = \text{ref}_0,\qquad q_{i+1} = q_i + R_z\!\left(\tfrac{\Psi_i + \Psi_{i+1}}{2}\right)(\text{ref}_{i+1} - \text{ref}_i + U_i),\qquad U_i = \textstyle\int u $$

So with the variables at the frozen copy, a knot turns the steps of the trajectory, each by the drift accumulated up
to it, and shifts them by the drift velocity; what the variables differ from the copy by is turned with the rig.

This is a **gauge**: for every value of the knots there are variables with the same true states, so the knots
over-parametrize the problem without changing it (same optimum, same cost). What changes is the path of the
optimizer: a mid-recording drift is one damped variable, not every later pose fighting the damping.

## How the factors see it

A factor that relates two rigs A and B (an inertial factor between consecutive rigs; a visual factor between the
host of a point and the rig that observes it, [host_points.md](host_points.md)) does not change when both are turned
about the vertical and shifted together. So it is evaluated with A as it is and B taken through
$H_A^{-1} H_B$ ($H_i$ the transform above):

$$ R_B \to R_z(d)\,R_B,\quad v_B \to R_z(d)\,v_B,\quad p_B \to \text{ref}_A + m + R_z(d)(p_B - \text{ref}_B) $$
$$ d = \Psi_B - \Psi_A,\qquad m = R_z(-\Psi_A)(q_B - q_A) $$

$d$ and $m$ depend only on the knots between the two times. The knot intervals are at least `--yaw-knot-sec` long
and follow the point tracks (longer where a track needs it) so that **no track spans more than two intervals**; a
factor then depends on at most two knots (`YawDriftPath::knotPairFor`), and the banded structure is kept. A factor of
one rig alone that uses its attitude and velocity (rolling shutter, row time) needs no wrapping: velocities are only
turned, so the body sees the same gravity and velocity whatever the knots. The wrapper appends the two knots as
variables of the wrapped factor and has exact Jacobians.

**Folding.** The knots are put into the variables before every optimizer iteration (`yawKnots_apply`: variables =
true states, knots = 0, frozen copy taken again). Left to accumulate over a call they take the variables far from
the copy, the problem becomes very non-linear and steps get rejected (a defect found by the tests and fixed). A
second defect, also fixed: with the row-time model the velocities were shifted as well as turned, which broke the
neutrality.

**Tests.** Neutrality (`VIBA_YAW_GAUGE_TEST=<rad/s>`): knots at +-3.4 deg/min and +-30 mm/s with compensating
variables give identical cost histograms. `--verify-jacobians` with `VIBA_YAW_KNOT_INIT` (knots started away from 0)
checks the wrapped Jacobians. The log reports the number of knot intervals longer than 40, 60 and 120 s.

## Results

Injected heading drift of 1.5 deg over the middle of the recording, error left at the end:

- **R_11_5cp** (1.5 deg over 160 s): without knots nothing moves for 6 iterations, then 13 iterations to converge;
  with the damping held at 1e-6 it is never corrected; **with knots it is gone after 1 iteration** (0.023 deg, then
  0.002).
- **R_12_10cp** (1.5 deg over 300 s, from a state 1.1 deg short): without knots 33 iterations, 1159 s; with knots
  12 iterations, 384 s; the same solution and final cost either way. Without knots the gap closes by 17 % per
  iteration once the damping reaches 1e-10.

On real data, from a filter trajectory, two passes with the scale-knot stage ([scale_knots.md](scale_knots.md)):

| recording | iterations with knots | without | score % with knots, pass 1 / pass 2 |
|---|---|---|---|
| R_11_5cp | 18 | 23 | 98.8 / 98.4 |
| sequence_1_19 | 16 | 24 | 93.9 / 97.5 |
| R_12_10cp | 18 | 47 | 68.5 / 68.2 |

Without knots the scores are the same within 0.5 points (sequence_1_19: 93.9 / 97.3), the final costs are equal to
the unit (R_12_10cp: 481 163 against 481 162) and no step was rejected: the knots are neutral, the gain is speed.

**Scale too?** A scale-drift rate could be added to the knots in the same way, but scale is observable to the IMU
over a minute (about 0.1 %, [scale_knots.md](scale_knots.md)) while heading and horizontal position are not observed
at all by an open-loop problem; the default keeps the scale-knot stage plus the 4-DoF drift knots.
