# VI-BA design notes

What was changed in VI-BA's optimizer and sensor model while refining trajectories of Project Aria recordings (the
[LaMAria](https://github.com/cvg/lamaria) benchmark, scored on its control points), why, and what each change
measured. One note per topic; the summaries below give the gist and point to them.

The common thread: **the optimizer was hiding the model**. Levenberg-Marquardt's damping held back the corrections
that matter most on long recordings (slow drifts of heading, position and scale), and far points froze the heading.
Once VI-BA converges fast and reliably, what is left is the information in the observations, and loop closure can
be added without giving up the banded direct solver.

## 1. Optimizer

**[Damping as a hidden prior](damping.md).** The damping $H_{ii} \leftarrow H_{ii}(1+\lambda) + \lambda$ is a
temporary prior on every variable. Its absolute part ($\lambda$, up to $10^{-5}$, never below $10^{-9}$ in the stock
schedule) is many orders above the curvature of a far point stored in world coordinates ($\sim10^{-22}$ at
$10^{13}$ m), so a few hundred such points could not move, and they pinned the heading of the whole trajectory
(R_12_10cp: 40 % where 82 % was reachable). Ceres (SLAM Dunk's batch) has a purely relative damping that falls to
$\sim10^{-22}$ and does not get stuck. A slow drift moves every later pose and is charged by the damping pose by pose,
so it waits until the damping is tiny. The stock schedule was also too cautious (every step gained 150-198 % of the
predicted reduction): the new defaults (x0.1 per good step, floor 1e-11, later passes from 1e-9, tolerances counted at
the floor, stop when no key rig moves by 0.5 mm) take R_11_5cp from 71 to 24 iterations, same solution.

**[Points in a host rig](host_points.md)** (`--point-param hostmix`, default). Each point lives in the body frame of
the rig of its middle observation: as a position (cartesian) when near, as direction and inverse depth beyond 20 m.
A point at infinity is an ordinary value that turns with its host and holds nothing in the world. Same cost
function; every run now converges by VI-BA's own test (none did with world points), R_12_10cp 40.5 -> 76.2 %.

**[Scale knots](scale_knots.md)** (`--scale-knot-sec 60`). One factor per minute, interpolated, multiplies gravity
and the integrated accelerometer signal; used for one iteration at the start of each pass, then the trajectory is
divided by it and the knots reset. A slow scale drift is read in one step instead of 20 iterations (+-0.25 % ramp:
0.031 -> 0.010 % rms left after 10 iterations). Gravity alone does not carry the scale.

**[Drift knots, 4 DoF](drift_knots.md)** (`--yaw-knot-sec 30`). Each knot holds a yaw rate and a position-drift
velocity on an interval of at least 30 s that follows the tracks (no track spans more than two). The true states
are the variables turned and shifted by the accumulated drift: an exact gauge, so the optimum and the cost are
unchanged, but a mid-recording drift is one variable instead of every later pose. Folded into the trajectory before
every iteration. An injected 1.5 deg heading drift goes in 1 iteration instead of 13-33; real data reach the same
optimum in 16-18 iterations instead of 23-47.

**[Elimination order](elimination_order.md)** (`--elimination-order time`, default). The reduced system is a band in
time, but BaSpaCho always ran AMD, which degenerates when the row-time model adds the velocity to every observation
(R_11_5cp: factor 4.3 GB, 3 min 15 s per factorization). BaSpaCho now accepts a given order
(`Settings::bottomPermutation`), small_thing sorts the variables by a key (the key-rig time; calibration windows and
knots after the last rig they couple), and the same run takes 1.7 GB and 28 s per factorization, 28 min instead of
77, with identical iterates.

## 2. Loop closure

**[PCG with the banded factor as preconditioner](loop_closure.md)** (`--lc-tracks` / `--lc-matches`). Loop
landmarks join rigs minutes apart and would fill the band; a pose graph followed by open-loop BA is pulled back by
the open-loop optimum. So the loop terms stay in the cost but out of the factorization: one variable per loop
landmark with all its observations, in a matrix-free store; BaSpaCho factorizes the banded open-loop matrix (plus the
curvature of per-visit virtual points) and that factor preconditions PCG on the full loop-closed reduced system, whose
product is $L(L^\top v)$ plus the loop terms. Before it, a scale-free pose graph (5-point verification, GNC per loop
pair) over key-rig poses or drift knots gives the starting correction, mapped back through the drift-knot transform.
On LaMAria's sequence_1_20 with SLAM Dunk's own measurements, VI-BA reaches 98.3 % from SLAM Dunk's pose graph and
97.3 % from its VIO before the pose graph, against 91.1 % for SLAM Dunk's loop-closed batch.

## 3. Sensor model

**[Parasitic light sensitivity and IMU timing](parasitic_light.md)** (Aria profile). The Aria SLAM cameras are global
shutter, but during the readout the stored charge keeps collecting light with sensitivity 1/89, so row $r$ is seen
later by $D = \varepsilon w (t_e + w) / (2 (t_e + \varepsilon w))$, $w = 0.35\,\text{ms} + r \cdot 19\,\mu\text{s}$,
attenuated by the smear of the ghost; each observation gets that time shift and a rolling-shutter factor on the
rig's velocity. With it, VI-BA's free estimate of the IMU time offsets lands 0.03 ms from the prediction of the
half-sample hold (factory offsets are instantaneous, VI-BA holds a sample over the interval ending at it), and a
freely estimated readout settles at the model's 1.7-1.9 ms. Cost -12.4 % on R_11_5cp; the effect on scores depends
on the recording.

**[Rotation of the Earth](earth_rotation.md)** (`--earth-rate-up`, Aria profile 5.365e-5 rad/s). The gyroscope
measures rotation against inertial space, the world turns with the Earth: the vertical component
$\Omega \sin\varphi$ (0.184 deg/min in Zurich) enters every inertial factor. It moves the heading rate by about
-0.02 deg/min; better scores on three of four recordings.

## Device profiles

The Aria-specific parts (row-time model, half-sample IMU timing, Earth rate at LaMAria's latitude) are set by
`--profile aria`, chosen automatically when the factory calibration has the Aria SLAM cameras; `generic` (e.g. TUM-VI)
turns them off. The optimizer changes of section 1 and the loop closure apply to every device.
