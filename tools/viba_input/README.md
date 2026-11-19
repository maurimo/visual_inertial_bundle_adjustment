# VIBA inputs from other datasets and map formats

Converters that write a VIBA input folder (format: see the top-level README and `vibaio.py`), for
visual-inertial recordings in the ASL layout (EuRoC, TUM-VI: `mav0/imu0/data.csv`, `mav0/camX/data.csv`).

| script | source | initial trajectory | observations |
|---|---|---|---|
| `../slamdunk_to_viba/slamdunk_to_viba.py` | a SLAM Dunk run | the state its global BA starts from, or its result | its keyframe observations (SIFT + KLT) |
| `asl_tracks_to_viba.py` | feature tracks of `klt_frontend.py` | any trajectory file (TUM format, EuRoC ground truth csv, ...) | the tracks, mapped back to measured pixels |
| `colmap_to_viba.py` | a COLMAP sparse model (text or binary) | the model's poses, made metric and gravity aligned (IMU alignment, Sim(3) to a trajectory, or as is) | the 2d tracks of the model's 3d points |

All converters

* take the camera-IMU calibration from the dataset (TUM-VI `dso/camchain.yaml`, EuRoC `mav0/cam*/sensor.yaml`)
  and the IMU noise from its Kalibr imu yaml; `--calib` replaces the cameras with another calibration:
  Kalibr camchain (`pinhole-equidistant`, `pinhole-radtan`, `ds`, `eucm`, `omni`) or Basalt json (`kb4`, `ds`,
  `eucm`, `ucm`, `pinhole`), eg `~/vio_ref/basalt/data/{tumvi_512,euroc}_{ds,eucm}_calib.json`. The
  observations do not depend on the camera model;
* shift the IMU timestamps by half a sample period (`--imu-stamp-shift-ms`), as VIBA's preintegration holds each
  sample over the period preceding its timestamp;
* write the dataset timestamp in `utc_timestamp_ns`, so `../slamdunk_to_viba/export_trajectories.py viba` gives
  trajectories in the dataset clock (`t[s] x y z qx qy qz qw`, IMU pose).

Other tools:

* `check_viba_input.py <input>`: triangulates a sample of points and prints reprojection errors per camera,
  without running VIBA. Errors of many pixels mean wrong extrinsics, pixel conventions, camera order or
  timestamps. Use `--max-span-kf` to limit the effect of drift of the initial trajectory.
* `viba_to_colmap.py <input> <out>`: the keyframes, cameras and triangulated points of an input as a COLMAP
  text model (for COLMAP's GUI, or to test `colmap_to_viba.py` with `--sim3`).
* `run_experiments.sh prepare|run|eval|list [regex]`: prepares, runs and evaluates the experiment set
  (SLAM Dunk / KLT tracks x TUM-VI / EuRoC x camera models), see its header. `prepare` runs no VIBA.

## Verification done (2026-11-19)

* `slamdunk_to_viba.py` after the refactoring onto `vibaio.py`: byte-identical outputs to the previous version
  (room1/run2 and slides1/full2, both initializations).
* Calibrations: Basalt's DS and EUCM calibrations of TUM-VI project within 0.4 px of Kalibr's Kannala-Brandt
  (same ray), extrinsics within 2 mm and 0.03 deg.
* Track files store undistorted normalized coordinates; for radtan cameras the tracker's `cv2.undistortPoints`
  stops after 5 iterations (errors up to 0.3 px at the border of EuRoC images). `vibaio.pixels_from_normalized`
  inverts that map and recovers the measured pixels to 3e-6 px (`--opencv-undistort-iters 0` disables it).
* `check_viba_input.py`, median reprojection error: SLAM Dunk final state on room1 0.23 px (KB), 0.25 px (DS);
  room1 tracks with Basalt's trajectory 0.7 px, with motion capture 0.85 px; EuRoC V1_01 tracks with ground truth
  0.6 px over 3 keyframes, growing with the track span (ground truth / track drift), smallest at zero time shift.
* COLMAP round trip (room1 SLAM Dunk final -> `viba_to_colmap.py --sim3` with scale 0.37 -> `colmap_to_viba.py`):
  IMU alignment recovers the scale to 0.3 %, gravity to 0.3 deg, velocities to 0.016 m/s; Sim(3) alignment to
  motion capture and `--align none` work too. The text model loads in pycolmap 4.2.1 (mean reprojection error
  0.34 px with COLMAP's own models), and the binary reader matches pycolmap's binary output exactly.
