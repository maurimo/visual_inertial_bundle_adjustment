# Points expressed in a host rig (cartesian or inverse depth)

`--point-param world|host|hostinv|hostmix` (default `hostmix`), `--point-far-dist` (default 20 m).
Code: `viba/problem/VisualFactorRelative.h`, `SingleSessionAdapter::setPointHosts` (`viba/single_session/VisualFactors.cpp`).

## Why

With points as world coordinates:

- a joint motion of a stretch of trajectory and of the points it sees (a slow change of heading along a recording)
  has to move every one of those points, and the damping of the points resists it;
- a low-parallax point, which the optimizer pushes ever farther, would have to move sideways by its distance times
  the rotation. Its curvature in that direction (about $10^{-22}$ at $10^{13}$ m) is far below the damping, so it
  stays, and it pins the absolute heading of every camera that sees it ([damping.md](damping.md)).

## The parametrization

Each point is expressed in the body frame of a **host rig**, the rig of the middle observation of its track (one of
the rigs that observe it). Two forms:

- **cartesian** (`host`): the position $x$ of the point in the host body frame;
- **inverse depth** (`hostinv`): $(u, v, \rho)$ with position $R_0\,[u, v, 1]^\top / \rho$, where $R_0$ is a fixed
  rotation whose third axis is the initial direction of the point in the host;
- **mixed** (`hostmix`, the default): inverse depth for points farther than `--point-far-dist` from the host,
  cartesian for the nearer ones.

All the geometry is done on $w \cdot x$ with $w = 1$ (cartesian) or $w = \rho$ (inverse depth), normalized only
before the projection. $\rho \le 0$ is not an error: $w \cdot x$ stays a valid direction through $\rho = 0$ (the
point passes through infinity); an observation fails only when that direction is behind the camera. So **a point at
infinity is an ordinary value**: when its host turns, the point turns with it, and it holds nothing in the world.

The visual factors for hosted points (`VisualFactorRelative.h`) take the host pose and the observing pose; an
observation from the host itself depends on the host only (and on the camera extrinsics and intrinsics). The
rolling-shutter / time-shifted versions (used by the row-time model, [parasitic_light.md](parasitic_light.md)) also
take the observing rig's velocity. The cost function is the same as with world points: on R_12_10cp the initial cost
is identical (777 972), and the Jacobians of the new factors agree with finite differences (`--verify-jacobians`,
`viba/problem/tests/TestVisualFactorRelative.cpp`).

With drift knots ([drift_knots.md](drift_knots.md)) an observation relates two rigs, host and observer, so it is
wrapped like an inertial factor; points in the host stay in their host when the knots are folded into the trajectory.

## Results (LaMAria, SLAM Dunk's tracks)

R_12_10cp, VI-BA started from a filter trajectory on SLAM Dunk's SIFT tracks (1449 key frames, 571 826
observations); score = LaMAria control-point score:

| variant | score % (ATE m) | iterations |
|---|---|---|
| stock (world points) | 40.9 (2.50) | 60, stuck |
| absolute part of the damping removed | 58.7 (1.22) | 100, still moving |
| world points brought back to 1000 km before every step | 68.3 (0.85) | 100, still moving |
| both | 69.2 (0.77) | 100 |
| world points not damped at all | 77.9 (0.51) | 100, not converged, fragile |
| **inverse depth in a host rig**, damping from 1e-7 | 69.3 (0.76) | **28, converged** |
| VI-BA started from SLAM Dunk's solution | 68.7 (0.91) | |

From the filter trajectory the optimizer now reaches the solution it reaches from SLAM Dunk's (1 cm apart). The
77.9 % run is an unconverged state with a higher cost.

With VI-BA's stock settings and only the point form changed: SIFT tracks from the filter trajectory 40.5 -> 76.2 %
(60 iterations stuck -> 39, converged); SLAM Dunk's standard input 79.6 % (250-iteration limit) -> 80.5 % (209,
converged). Across six training recordings on the standard input every host-rig run converges by VI-BA's own test
and no world-point run does (mean score 90.5 % world, 90.2 % mixed):

| recording | world points | host rig, mixed |
|---|---|---|
| sequence_1_19 | 94.0 (250 limit) | 95.0 (121) |
| sequence_1_20 | 90.5 (250 limit) | 92.8 (161) |
| sequence_2_11 | 91.5 (250 limit) | 93.2 (102) |
| sequence_4_11 | 92.5 (250 limit) | 85.0 (129) |
| R_11_5cp | 94.6 (250 limit) | 94.6 (165) |
| R_12_10cp | 79.6 (250 limit) | 80.5 (209) |

**Open: sequence_4_11.** Five host-rig variants (mixed at 5, 20, 80 m; all inverse depth; all cartesian) end within
7 mm of each other at 85.0-85.2 %, all with a lower cost than the world-point run; the 92.5 % is SLAM Dunk's
starting trajectory, which the stuck optimizer kept. VI-BA's cost prefers a heading profile 0.6 deg off there: the
cause is in the model, not in the optimizer.
