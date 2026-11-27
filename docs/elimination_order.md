# Elimination order: time order instead of AMD

`--elimination-order time|amd` (default `time`). Code: `Optimizer::initSolver` (`lib/small_thing/Optimizer.cpp`),
`VarBase::orderingKey` (`lib/small_thing/Variable.h`), BaSpaCho `Settings::bottomPermutation` (branch
`mau/fixed-order` of the BaSpaCho submodule).

## The structure of the problem

The direct solver eliminates the points first (Schur complement, BaSpaCho's sparse elimination ranges), then
factorizes the **reduced system** of the key rigs (pose, velocity, angular velocity), the calibration windows (IMU,
cameras, extrinsics), the knots and the global variables. Every factor of the open-loop problem joins variables
close in time: inertial factors join consecutive rigs, and a point joins the rigs that see it, over the length of its
track (with drift knots no track spans more than two knot intervals, [drift_knots.md](drift_knots.md)). Loop
observations are kept out of the factorized matrix ([loop_closure.md](loop_closure.md)). So the reduced system is a
**band in time**, and the natural elimination order is known in advance: by time.

## What BaSpaCho did

BaSpaCho had no way to receive an order: `createSolver` always ran AMD (approximate minimum degree, Eigen's
implementation) on the reduced block structure, then reordered along the elimination tree and merged supernodes.
AMD does well on many structures, but it can fail badly here. On LaMAria's R_11_5cp (4761 key rigs, 180 054 points,
192 camera and 96 IMU calibration windows), turning on the parasitic-light row-time model
([parasitic_light.md](parasitic_light.md)), which adds the rig's velocity to every observation, made the factor grow
from 1.1 to 4.3 GB and each factorization from 13 s to 3 min 15 s; the run took 77 min instead of 12.5.

The likely cause (not verified): Eigen's AMD sets aside every node with more than $\max(16, 10\sqrt n)$ neighbours
as *dense* and orders those last, without fill reduction. Here $n \approx 15\,000$ block variables, so the threshold
is about 1200 neighbours. A key rig that sees points tracked over up to a minute is near it, and the velocity
coupling of the row-time model (each point's clique grows from 6 to 9 dimensions per rig) pushes many over: they end
up in one large dense block.

## The change

- **BaSpaCho** (`mau/fixed-order`, one commit): `Settings::bottomPermutation`, the elimination order of the
  parameters after the sparse elimination ranges; when empty, AMD as before. The elimination-tree postorder (which
  adds no fill) and the parameters forced last are applied as usual.
- **small_thing**: each variable has an `orderingKey` (NaN by default) and `Optimizer::Settings::eliminationOrder`
  is `Order_Amd` or `Order_Keyed`. With `Order_Keyed` the reduced variables are sorted by key (ties by index); a
  variable without a key takes the largest key among the variables it shares a block with, so a calibration window
  or a knot goes right after the last key rig it couples, and a variable coupled to no keyed variable goes last.
- **VI-BA** keys the pose, velocity and angular velocity of each key rig by its timestamp; `--elimination-order
  time` (default) selects `Order_Keyed`, `amd` the old behaviour.

## Results (osiris, 4 threads)

R_11_5cp, Aria profile, 2 passes of at most 50 iterations, 18 factorizations each way:

| | factor | factorization | solve | iteration | run | peak memory | score |
|---|---|---|---|---|---|---|---|
| row-time model on, amd | 4.3 GB | 3 min 15 s | 1.25 s | 3 min 22 s | 76.9 min | 6.8 GB | 97.4 % |
| row-time model on, **time** | **1.7 GB** | **28 s** | **0.60 s** | **40 s** | **27.8 min** | **3.9 GB** | 97.4 % |
| row-time model off, amd | 1.1 GB | 13.4 s | 0.42 s | | 12.5 min | | 98.1 % |
| row-time model off, time | 1.3 GB | 15.0 s | 0.42 s | | | | |

The iterates are the same (the costs agree to every printed digit, the final trajectory and score are identical):
the order only changes the speed. Where AMD is not hit by the dense-node problem, time order is somewhat slower:

| | factor amd / time | factorization amd / time | run amd / time | result |
|---|---|---|---|---|
| TUM-VI room1, generic profile | 75 / 87 MB | 1.05 / 1.2 s | 35 / 37 s | same final cost |
| LaMAria sequence_1_20, SLAM Dunk input, loop closure | 0.35 / 0.44 GB | 2.1-2.7 / 3.25 s | 406 / 482 s | 98.3 % both |

Time order is the default because its worst case is a modest slowdown and AMD's is a factor of 7. Its memory grows
linearly with the length of the recording (for a given track length), which is what makes the direct solver usable
on long recordings ([damping.md](damping.md): no automatic switch to an iterative solver).
