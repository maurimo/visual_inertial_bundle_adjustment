#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Convert feature tracks on an ASL dataset (EuRoC, TUM-VI) plus an initial trajectory into a VIBA
input folder.

Inputs:
  --dataset   ASL folder with mav0/ (IMU, image timestamps) and its calibration: TUM-VI's
              dso/camchain.yaml + dso/imu_config.yaml, or EuRoC's mav0/cam*/sensor.yaml +
              mav0/imu0/sensor.yaml
  --tracks    track file of klt_frontend.py (sliding_marg/real): a header (S, E, I, N, E1, C lines),
              then per frame `F <t[s]> <n_imu> <n_obs>`, n_imu IMU rows, n_obs rows
              `<track id> [<camera>] <x> <y>` in undistorted normalized coordinates
  --traj      initial T_world_imu trajectory (world gravity aligned, metric), formats of
              vibaio.read_trajectory: TUM `t x y z qx qy qz qw` (Basalt, OpenVINS, DM-VIO, SLAM Dunk
              exports...), the 21-column outputs of sliding_marg/real, or EuRoC ground truth csv

The normalized coordinates are mapped back to the measured pixels with the dataset calibration
the tracker used (inverting OpenCV's 5-iteration cv2.undistortPoints for radtan cameras, see
vibaio.pixels_from_normalized). The camera model given to VIBA is the dataset's, or --calib
(Kalibr camchain .yaml, Basalt .json: Double Sphere / EUCM / UCM / KB4 / pinhole).

Example (TUM-VI room1, Basalt's VIO as initialization, Double Sphere model):
  R=~/git4/experiments/msckf_curvature_cpp/causal/sliding_marg/real
  asl_tracks_to_viba.py --dataset /galway/data/tumvi/dataset-room1_512_16 \\
      --tracks $R/data/tumvi/room1anc_st.txt --traj $R/out/cmp/room1/ref_basalt.txt \\
      --calib ~/vio_ref/basalt/data/tumvi_512_ds_calib.json --out /tmp/viba_room1
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vibaio  # noqa: E402


def read_tracks(path):
    """Returns (frame times [ns] list, list of (ids, cams, xy) per frame, header dict)."""
    header, times, frames = {}, [], []
    with open(path) as f:
        lines = iter(f)
        for line in lines:
            tok = line.split()
            if not tok:
                continue
            if tok[0] != "F":
                header[tok[0]] = tok[1:]
                continue
            stereo = header.get("C", ["1"])[0] != "1"
            t_ns = vibaio.parse_stamp_ns(tok[1])
            n_imu, n_obs = int(tok[2]), int(tok[3])
            for _ in range(n_imu):
                next(lines)
            rows = [next(lines).split() for _ in range(n_obs)]
            if stereo:
                ids = np.array([int(r[0]) for r in rows], dtype=np.int64)
                cams = np.array([int(r[1]) for r in rows], dtype=int)
                xy = np.array([[float(r[2]), float(r[3])] for r in rows]).reshape(-1, 2)
            else:
                ids = np.array([int(r[0]) for r in rows], dtype=np.int64)
                cams = np.zeros(len(rows), dtype=int)
                xy = np.array([[float(r[1]), float(r[2])] for r in rows]).reshape(-1, 2)
            times.append(t_ns)
            frames.append((ids, cams, xy))
    return np.array(times, dtype=np.int64), frames, header


def select_keyframes(t_ns, min_dt_ns):
    keep, last = [], None
    for i, t in enumerate(t_ns):
        if last is None or t - last >= min_dt_ns - 1_000_000:  # 1 ms tolerance for frame jitter
            keep.append(i)
            last = t
    return np.array(keep, dtype=int)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, type=Path, help="ASL dataset folder (with mav0/)")
    ap.add_argument("--tracks", required=True, type=Path, help="track file (klt_frontend.py format)")
    ap.add_argument("--traj", required=True, type=Path, help="initial trajectory (T_world_imu)")
    ap.add_argument("--traj-format", default="auto", choices=["auto", "tum", "ours", "asl"])
    ap.add_argument("--out", required=True, type=Path, help="output folder (VIBA input)")
    ap.add_argument("--calib", type=Path, default=None,
                    help="camera calibration for VIBA replacing the dataset's (Kalibr .yaml, Basalt .json)")
    ap.add_argument("--track-cams", default="cam0,cam1",
                    help="dataset cameras of the track file's camera indices 0, 1 (default cam0,cam1)")
    ap.add_argument("--kf-dt", type=float, default=0.0,
                    help="minimum time between keyframes [s] (default 0: every frame of the track file)")
    ap.add_argument("--min-track-kf", type=int, default=2, help="drop tracks seen in fewer keyframes")
    ap.add_argument("--sigma-px", type=float, default=0.5, help="observation std [px]")
    ap.add_argument("--opencv-undistort-iters", type=int, default=5,
                    help="iterations of the tracker's cv2.undistortPoints (radtan cameras), 0: exact")
    ap.add_argument("--valid-radius", type=float, default=None,
                    help="valid image radius [px] (default: 300 on TUM-VI 512, none otherwise)")
    ap.add_argument("--imu-stamp-shift-ms", type=float, default=None,
                    help="added to IMU timestamps (default: half the median sample period; 0 disables)")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    mav0 = args.dataset / "mav0"

    # ---- calibration
    data_cams, noise = vibaio.dataset_calibration(args.dataset)
    by_label = {c.label: c for c in data_cams}
    track_cams = args.track_cams.split(",")
    for lbl in track_cams:
        if lbl not in by_label:
            raise SystemExit(f"camera {lbl} not in the dataset calibration ({list(by_label)})")
    cams = list(data_cams)
    if args.calib is not None:
        cams, _, _ = vibaio.override_calibration(cams, args.calib)
    valid_radius = args.valid_radius if args.valid_radius is not None else vibaio.default_valid_radius(
        args.dataset, data_cams)
    for c in cams:
        c.valid_radius = valid_radius
    cam_index = {c.label: i for i, c in enumerate(cams)}

    # ---- IMU (clock: dataset ns minus an offset, so that VIBA's timestamps stay small)
    t_imu, gyro, accel = vibaio.read_asl_imu(mav0)
    offset_ns = int(t_imu[0] // 10**9 - 1) * 10**9
    shift_ns = vibaio.imu_stamp_shift_ns(t_imu, args.imu_stamp_shift_ms)
    t_imu_viba = t_imu - offset_ns + shift_ns
    vibaio.write_imu(out, t_imu_viba, gyro, accel)
    vibaio.write_imu_noise(out, noise)

    # ---- frames: snap to image timestamps (the track file has float seconds)
    t_frames, frames, header = read_tracks(args.tracks)
    t_img = vibaio.read_asl_image_stamps(mav0, track_cams[0])
    j = np.clip(np.searchsorted(t_img, t_frames), 1, len(t_img) - 1)
    j -= (t_frames - t_img[j - 1]) < (t_img[j] - t_frames)
    snap_err = np.abs(t_img[j] - t_frames)
    if snap_err.max() > 1_000_000:
        raise SystemExit(f"track frames do not match image timestamps (max {snap_err.max()} ns)")
    t_frames = t_img[j]

    # ---- keyframes: inside the trajectory and the IMU range
    t_tr, p_tr, q_tr, v_tr = vibaio.read_trajectory(args.traj, args.traj_format)
    kf = select_keyframes(t_frames, int(args.kf_dt * 1e9))
    t_kf = t_frames[kf]
    p_kf, q_kf, valid = vibaio.interpolate_poses(t_tr, p_tr, q_tr, t_kf)
    margin_ns = 10_000_000
    valid &= (t_kf - offset_ns > t_imu_viba[0] + margin_ns) & (t_kf - offset_ns < t_imu_viba[-1] - margin_ns)
    if not valid.any():
        raise SystemExit("no keyframe inside the trajectory and IMU time ranges")
    kf, t_kf, p_kf, q_kf = kf[valid], t_kf[valid], p_kf[valid], q_kf[valid]
    v_kf = vibaio.velocities(t_tr, p_tr, q_tr, t_kf, v_tr)
    omega = vibaio.nearest_gyro(t_imu, gyro, t_kf)
    kf_us = (t_kf - offset_ns) // 1000
    assert len(np.unique(kf_us)) == len(kf_us)
    vibaio.write_trajectory(out, kf_us, t_kf, p_kf, q_kf, v_kf, omega, session="tracks")
    vibaio.write_calibration(out, cams, vibaio.Imu("imu0"), kf_us, t_kf,
                             "TUM-VI" if vibaio.is_tumvi(args.dataset) else "ASL")

    # ---- observations (measured pixels, dataset calibration of the tracker)
    pid, kfi, cam, uv = [], [], [], []
    for k, fi in enumerate(kf):
        ids, cidx, xy = frames[fi]
        for c in np.unique(cidx):
            m = cidx == c
            label = track_cams[c]
            uv.append(vibaio.pixels_from_normalized(by_label[label], xy[m], args.opencv_undistort_iters))
            pid.append(ids[m])
            kfi.append(np.full(m.sum(), k))
            cam.append(np.full(m.sum(), cam_index[label]))
    pid, kfi, cam, uv = np.concatenate(pid), np.concatenate(kfi), np.concatenate(cam), np.concatenate(uv)
    good = vibaio.filter_tracks(pid, kfi, args.min_track_kf)
    vibaio.write_observations(out, pid[good], kf_us[kfi[good]], cam[good], uv[good], 1.0 / args.sigma_px)

    print(f"{out}: {len(kf)} keyframes of {len(t_frames)} frames, {len(np.unique(pid[good]))} points, "
          f"{good.sum()} observations, {len(t_imu)} IMU samples (stamp shift {shift_ns / 1e6:.3f} ms), "
          f"cameras {[c.model for c in cams]}, clock offset {offset_ns} ns")


if __name__ == "__main__":
    main()
