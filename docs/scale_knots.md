# Scale knots: reading a slow scale drift in one iteration

`--scale-knot-sec` (default 60; 0: off), `--scale-knot-std` (0.05), `--scale-knot-rw-std` (0.003),
`--scale-knot-iterations` (1), `--scale-knot-stages` (1).
Code: `SingleSessionProblem::scaleKnots_init` / `scaleKnots_apply` (`viba/problem/SingleSessionProblem.cpp`),
`SingleSessionAdapter::applyScaleKnots`.

## The problem

A slow scale drift of the trajectory (the start trajectory of a filter is a little too large in one minute, a little
too small in the next) is a soft mode: the data hardly sees it, and correcting it moves every pose
([damping.md](damping.md)). A test with a known answer: a smooth +-0.25 % scale ramp put into VI-BA's converged
solution on R_11_5cp, then 10 iterations. Iteration 1 removes the cost (119 622 -> 111 570, reference 111 563) and
the calibration jumps back, but 98 % of the drift stays in the trajectory: a drifted trajectory with a correct
calibration costs 13 units out of 111 560. The drift goes only between iterations 5 and about 20, as the damping
comes down to its floor; after 10 iterations 0.031 % rms is left.

## What carries the scale (and why gravity knots did not work)

A first idea was to let the gravity magnitude vary in time (one knot every 60 s), so that gravity "encodes" the
scale, and straighten it afterwards. It did worse (0.094-0.100 % rms left against 0.031 %; the knots stayed within
0.08 % of 9.81). With the accelerometer calibration where it belongs, a trajectory too large by $s$ leaves an
inertial residual of $(s - 1)$ times the *dynamic* velocity change: gravity is not coupled to the scale of the
trajectory, and a free gravity only trades against the vertical accelerometer bias. What scales together with the
trajectory (checked: with all of these and gravity multiplied by 1.01 every class of cost stays the same):
positions, velocities, points, lever arms, the accelerometer rectification (divided), gravity, the accelerometer
bias (left alone: cost 5247 -> 9433), and the scaling must be about the IMU, not the device origin (otherwise
5247 -> 7300).

## The scale knots

Knots $s_k$, one every `--scale-knot-sec` seconds, linearly interpolated in time, multiply the IMU's metric: gravity
**and** the integrated accelerometer signal of each inertial factor,

$$ e_v = s\,(\Delta V + R\,g\,\Delta t) - R\,(v_j - v_i) $$

(and likewise in the position residual), with a prior (`--scale-knot-std`) and a random walk between neighbouring
knots (`--scale-knot-rw-std`, 0.001-0.003 per minute). The IMU alone knows the scale of one minute only to about
0.1 %, so knots kept for the whole optimization fit noise. They are used in a **stage** instead:

1. at the start of each pass, `--scale-knot-iterations` (1) iterations with the knots as variables: the drift is read
   in one step, because a knot is one variable and not thousands of damped poses;
2. **straighten**: the steps of the IMU trajectory and the velocities are divided by $s(t)$, world points move with
   the rig of their middle observation, hosted points are rescaled in their host (inverse depth multiplied), and the
   knots are reset to 1;
3. the ordinary optimization follows (with the drift knots, [drift_knots.md](drift_knots.md)).

With `--scale-knot-iterations 0` the knots stay in for the whole optimization and are written to
`scale_knots.csv` (a diagnostic).

## Results

R_11_5cp, 10 iterations in all, scale error left (% rms) / distance to the reference:

| | +-0.25 % ramp | +-3 % ramp |
|---|---|---|
| ordinary iterations | 0.031 / 17 mm | 0.371 / 208 mm |
| with knot stages | **0.010 / 5 mm** | **0.064 / 24 mm** |
| control, no ramp | 0.009 / 2 mm | |

(+-3 %: 2 + 2 knot iterations, straightened after each, then 6 ordinary ones.) sequence_1_19, +-0.25 %: 0.030 ->
0.007 % rms, 33 -> 6 mm.

On real data, from a filter trajectory (two knot iterations, straighten, two more, straighten, one ordinary pass to
convergence): sequence_1_19 reaches 84.5 % in 4 + 119 iterations, where two ordinary passes take 250 iterations each
for 84.3 %; the knots read the filter's scale error (+0.1 ... +1.3 %, its trajectory is too large). On R_12_10cp
(-1.9 ... +0.9 %) the second pass corrects mostly the heading, which scale knots do not touch: that is what the
drift knots are for. The default schedule keeps both: a scale-knot stage at the start of each pass, then the drift
knots during the whole optimization.
