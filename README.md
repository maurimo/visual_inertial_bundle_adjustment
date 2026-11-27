[![CI](https://github.com/facebookresearch/visual_inertial_bundle_adjustment/actions/workflows/ci.yml/badge.svg)](https://github.com/facebookresearch/visual_inertial_bundle_adjustment/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/facebookresearch/visual_inertial_bundle_adjustment/blob/main/LICENSE)
[![C++](https://img.shields.io/badge/C++-17-blue.svg)](https://isocpp.org/)
[![PRs](https://img.shields.io/badge/PRs-welcome-green.svg)](https://github.com/facebookresearch/visual_inertial_bundle_adjustment/blob/main/CONTRIBUTING.md)

# Visual Inertial Bundle Adjustment

This is an implementation of a full sensor-modelling visual intertial bundle adjustment, designed to provide the
highest level of optimization of device states for Aria Gen1/Gen2 glasses. As a tool, it allows to reoptimize
estimates on recordings data. Other camera/IMU rigs can be used by providing the inputs in the same format (see
[Other devices](#other-devices-tum-vi-and-slam-dunk-results)). Since it works with a very dense problem formulation, you can expect to be able
to optimize up to 20m-30m of Aria recording.

![](https://github.com/facebookresearch/visual_inertial_bundle_adjustment/raw/refs/heads/main/docs/img/gui_animation.gif)

## How to run

Follow instrution below for building. Assuming the folder $VIBA_INPUT_DIR has input data, you can run:

```
build/interfaces/ark/ark_vi_ba -i $VIBA_INPUT_DIR -o $OUTPUT_DIR
```

A visualization of the optimization process can also be displayed on selected platforms (Linux-X11/Mac at this stage):

```
build/interfaces/ark/ark_vi_ba_gui -i $VIBA_INPUT_DIR -o $OUTPUT_DIR
```

The input data in the folder $VIBA_INPUT_DIR should be:

```
factory_calibration.json   # factory calibration, as extracted from VRS file
imu_samples_imu-left.csv   # IMU measurements (left-IMU), as extracted from VRS file
imu_samples_imu-right.csv  # IMU measurements (right-IMU), as extracted from VRS file
online_calibration.jsonl   # online calibration (obtained from MPS)
open_loop_trajectory.csv   # open-loop trajectory (obtained from MPS)
session_observations.csv   # recording observation (tracking points + SLAM loop-closing points)
vrs_source_info.json       # description of camera/imu layout during SLAM (indices in session_observations.csv)
imu_noise.json             # (optional) IMU noise model per IMU label, overriding the defaults (fitting Aria)
```

Notes on the formats:
* `session_observations.csv`: the column `capture_timestamp_ns` holds **microseconds** (as the tracking timestamps
  of the trajectory and online calibration). The name is kept for compatibility.
* `imu_noise.json`, example (scalars apply to all three axes, arrays give per-axis values):
  `{"imu0": {"accel_sample_variance": 1.57e-3, "gyro_sample_variance": 5.1e-6,
  "accel_bias_random_walk_variance_per_sec": 7.4e-7, "gyro_bias_random_walk_variance_per_sec": 4.8e-10}}`.
  Other supported fields: `accel_bias_turnon_std`, `gyro_bias_turnon_std`, `accel_scale_turnon_std`,
  `gyro_scale_turnon_std`.
* Camera models supported in calibration files (`Projection.Name`, parameters in `Projection.Params`):

  | name | parameters | as in |
  |---|---|---|
  | `FisheyeRadTanThinPrism` (Fisheye624) | f, cx, cy, k0..k5, p0, p1, s0..s3 | Aria |
  | `Fisheye62` | f, cx, cy, k0..k5, p0, p1 | Aria |
  | `KannalaBrandtK3` | fx, fy, cx, cy, k1..k4 | Kalibr `equidistant`, Basalt `kb4`, OpenCV fisheye |
  | `PinholeRadTan` | fx, fy, cx, cy, k1, k2, p1, p2 | Kalibr `radtan` (EuRoC), COLMAP `OPENCV` |
  | `PinholeRadTanK3` | fx, fy, cx, cy, k1, k2, p1, p2, k3 | OpenCV 5 coefficients |
  | `DoubleSphere` | fx, fy, cx, cy, xi, alpha | Basalt / Kalibr `ds` |
  | `ExtendedUnified` | fx, fy, cx, cy, alpha, beta | Basalt / Kalibr `eucm` |
  | `Unified` | fx, fy, cx, cy, alpha | Basalt `ucm` (Kalibr `omni` without distortion) |
  | `Linear` | fx, fy, cx, cy | pinhole |
  | `Spherical` | fx, fy, cx, cy | (no Jacobians, not usable for optimization) |

  Aliases are accepted (case insensitive): `Pinhole`, `KB4`, `KannalaBrandt`, `Equidistant`, `RadTan`, `DS`,
  `EUCM`, `UCM`. Pixel (0, 0) is the center of the top-left pixel.

While outputs in $OUTPUT_DIR are:

```
closed_loop_framerate_trajectory.csv  # re-estimated trajectory
online_calibration.jsonl              # re-estimated calibration
open_loop_framerate_trajectory.csv    # re-estimated trajectory (same data, in different format)
```

Notice that the files `closed_loop_framerate_trajectory.csv` and `open_loop_framerate_trajectory.csv` contain
the same data, just in different file format for convenience.

### From MPS data

All the data which are not coming from MPS can be recovered from VRS file and running Colmap pipeline.

* The inputs `factory_calibration.json` and `imu_samples_*.csv` can be extracted from a VRS file running
  (requires `pip install projectaria-tools`)

```
python3 tools/process_vrs/process_vrs.py -i $VRS -o $VIBA_INPUT_DIR
```

* The `session_observations.csv` and `vrs_source_info.json` can be obtained using the stripped-down Python library
  included in `tools/save_observations`. This library is derived from the [LaMAria project](https://github.com/cvg/lamaria)
  (CVG, ETH Zurich) and contains the minimum required code to process VRS files and generate observations for
  visual-inertial bundle adjustment.

  To use it, first install the dependencies (from the `tools/save_observations` directory):

```
pip install -r requirements.txt
```

  Additionally, you need to install the [VRS CLI tool](https://github.com/facebookresearch/vrs) and ensure the `vrs`
  command is available in your PATH.

  Then run (from `tools/save_observations`, assuming MPS's results are in `$MPS_DATA`):

```
python -m save_observations --output $VIBA_INPUT_DIR \
       --vrs $VRS \
       --mps-path $MPS_DATA
```

### Other devices: TUM-VI and SLAM Dunk results

`tools/slamdunk_to_viba` converts a run of SLAM Dunk (Van Meter and Paul, 2026) on a
[TUM-VI](https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset) sequence into VIBA's input (stereo
equidistant fisheye cameras, one IMU), initialized either from the state SLAM Dunk's global bundle adjustment starts
from, or from its result. `compare.sh` runs both and evaluates them against SLAM Dunk's own optimizer:

```
tools/slamdunk_to_viba/compare.sh room1 run2 /tmp/viba_cmp      # see the script for the paths it uses
```

`tools/viba_input` has converters for other sources (EuRoC / TUM-VI feature tracks with an initial trajectory,
COLMAP models with IMU data), a consistency check of VIBA inputs and a driver of experiments over datasets and
camera models, see [its README](tools/viba_input/README.md).

### Aria recordings (LaMAria): SLAM Dunk results

Other estimators' results on Project Aria Gen1 recordings (e.g. the [LaMAria](https://github.com/cvg/lamaria)
benchmark) can be refined by VIBA: `tools/slamdunk_to_viba` converts SLAM Dunk's (`aria_slamdunk_to_viba.py`,
`sd_start.py`), `tools/loop_closure` prepares the loop-closing points, `tools/aria_postprocess` turns VIBA's output
into a trajectory for evaluation. The inputs start from the VRS extraction of the recording (factory calibration,
raw `imu-right` samples):

```
python3 tools/process_vrs/process_vrs.py -i $VRS -o $VRS_OUT
```

The inputs they write have the Aria cameras `camera-slam-left/right`, so `--profile auto` (the default) picks the
Aria profile: parasitic-light row time, half-sample IMU timing, Earth rotation. The row-time model needs the
exposure of every image, `expo_cam0.csv` / `expo_cam1.csv` (lines `t_ns,exposure_s`, e.g. from the image records'
exposure in the VRS) in the input folder or in `--pls-dir`; without them it is off. The runs below used
`--num-threads 4 --num-passes 2 --max-num-iterations 50`.

**SLAM Dunk's result, refined** (its optimized key frames and final, outlier-filtered observations).
`$RUN` is a SLAM Dunk run folder (`estimate/work/...`), `$REC` the recording folder SLAM Dunk extracted:

```
python3 tools/slamdunk_to_viba/aria_slamdunk_to_viba.py --run $RUN --rec $REC --vrs $VRS_OUT --out $IN --init final
build/interfaces/ark/ark_vi_ba -i $IN -o $OUT
```

**SLAM Dunk's loop-closed input, with loop closure in VIBA.** `--init input` takes the state and the observations
SLAM Dunk's global bundle adjustment starts from: its pose-graph trajectory, and tracking points whose ids are
shared by the revisits (the loop-closing points). VIBA's open-loop problem is a band in time, so the points seen
again after a gap are split off and given to the loop closure (`--lc-tracks`, optionally `--lc-matches` for the
pose-graph stage; loop closure runs if and only if one of them is given):

```
python3 tools/slamdunk_to_viba/aria_slamdunk_to_viba.py --run $RUN --rec $REC --vrs $VRS_OUT --out $SD_IN --init input
python3 tools/loop_closure/lc_split_viba_input.py $SD_IN/session_observations.csv $SPLIT --gap 27
mkdir -p $IN && for f in $SD_IN/*; do ln -s $f $IN/; done
ln -sf $SPLIT/session_observations.csv $IN/session_observations.csv   # points without a gap of more than 27 s
build/interfaces/ark/ark_vi_ba -i $IN -o $OUT --lc-tracks $SPLIT/loop_tracks.csv --lc-matches $SPLIT/lc_matches.csv
```

To start instead from SLAM Dunk's VIO before its pose graph (or any other trajectory: lines `t_ns x y z qx qy qz
qw` of the IMU), `sd_start.py` fits that trajectory into the input's world and writes the starting states; the
rest of `$IN` is as above:

```
head -1 $SD_IN/online_calibration.jsonl > $SPLIT/calib1.json
python3 tools/slamdunk_to_viba/sd_start.py $SD_IN/open_loop_trajectory.csv $SPLIT/calib1.json $VIO_POSES \
        $SPLIT/open_loop_trajectory.csv --no-body-offset
ln -sf $SPLIT/open_loop_trajectory.csv $IN/open_loop_trajectory.csv
```

On LaMAria's `sequence_1_20` (21 min) both starts reach the same solution, 98.3 % and 97.3 % of the benchmark's
control-point score, against 91.1 % for SLAM Dunk's own loop-closed batch.

**Output for evaluation.** `postprocess.py` writes the optimized trajectory of the IMU at every camera frame between
the first and the last key frame (`traj.txt`, `t[s] x y z qx qy qz qw`, the IMU integrated with VIBA's calibration
between key frames), plus the key frames alone and the calibration over time:

```
python3 tools/aria_postprocess/postprocess.py $IN $OUT $REC/cam0/data.csv $DEST
```

## How to build

The only git submodule is `deps/baspacho`; populate it if needed:

```
git submodule update --init --recursive
```

All other dependencies (Eigen, Sophus, CLI11, nlohmann-json, dispenso, googletest, and for the GUI sokol, imgui and
implot) are downloaded by CMake. fmt is used from the system if found, otherwise downloaded. BLAS (OpenBLAS) is
required by BaSpaCho.

### Using Pixi package manager (Linux/Mac)

This is the recommended approach on Mac, and also works on Linux:

* Install pixi: <https://pixi.sh/latest/installation/>
Configure and build:

```
pixi run prepare
pixi run build
```

### Directly building with CMake (on Linux)

Install system deps, on Fedora:

```
dnf install fmt-devel openblas-devel
```

Install system deps, on Ubuntu:

```
sudo apt-get install -y libfmt-dev libopenblas-dev cmake build-essential
```

For the GUI app (`ark_vi_ba_gui`) also OpenGL and X11 development packages are needed (Ubuntu:
`libgl1-mesa-dev libx11-dev libxi-dev libxcursor-dev`).

Configure with CMake:

```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
```

Build:

```
cmake --build build -- -j32
```

(Optional) Run Tests:

```
ctest --test-dir build
```

## Development

Install deps to enable pre-commit git hooks and format/lint:

```
dnf install clang-tools-extra python3-pip
pip install --user pre-commit
```

## Description of the model

The design of the optimizer (damping, points in a host rig, scale and drift knots, elimination order), of the loop
closure and of the Aria sensor-model additions is described in the [design notes](docs/README.md).

The whole calibration state for all sensors (all intrinsics and extrinsics) is optimized, and all recording frames
are used with all visual observations.

* Inertial device poses state is formed by world pose, linear velocity and angular velocity.
* IMU calibration is fully supported (biases/scaling/non-orthogonality, plus all time offsets:
  IMU relative to tracking clock, and accel-to-gyro time offset)
* RGB camera is uses a rolling shutter aware modelling (camera time offset and readout time are optional
  variables of the optimization problem)
* Calibration's variation in time is modelled as a random walk, with windows of 5 seconds where it's assumed to be
  constant.
* All calibration state (intrinsics/extrinsics) have (weak) factory calibration priors which prevent the optimizer from
  setting them to unreasonable estimates in the portions of recordings where they are not well constrained by observations
  (for instance, in static recording portions).
* Multiple IMUs are supported, and the inertial poses are related to the estimate of instantaneous angular velocity
  (omega priors). Since the secondary IMU's extrinsics are optimization variables, this is required to prevent the
  unconstrained angular velocity from absorbing the secondary IMU's pose/velocity errors.
* Point tracks are expressed in a host rig (the rig of the middle observation of the track) rather than in the world:
  as inverse depth along their initial direction when farther than `--point-far-dist` (20m) from the host, as a
  position in the host's body frame otherwise (`--point-param hostmix`, the default; `host`, `hostinv` and `world`
  select a single form). A stretch of trajectory and the points it sees can then move together without the points
  changing, and a low-parallax point can sit at infinity: with world coordinates such points, pushed ever farther by
  the optimizer, cannot follow a change of heading and hold the trajectory at the heading it started with.
  Not available yet with rolling shutter or camera time offset (world coordinates are used then).

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

The camera projection models in `lib/calibration` are derived from
[projectaria_tools](https://github.com/facebookresearch/projectaria_tools) and keep its Apache 2.0 license headers.
