#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Convert a SLAM Dunk run (on an ASL dataset: TUM-VI, EuRoC) into a VIBA input folder.

SLAM Dunk (J. Van Meter, A. Paul) is an offline visual-inertial system whose last
stage is a global bundle adjustment over keyframes. This script reads the files of a finished run
(only its outputs, no code of it is used) and writes the files expected by `ark_vi_ba`:

  factory_calibration.json   nominal dataset calibration (device frame = IMU frame), or --calib
  online_calibration.jsonl   one entry per keyframe (same nominal calibration)
  open_loop_trajectory.csv   keyframe poses/velocities/angular velocities (initial guess)
  session_observations.csv   visual observations at keyframes (at the dataset's image resolution)
  imu_samples_imu0.csv       dataset IMU samples
  vrs_source_info.json       camera/IMU layout
  imu_noise.json             dataset IMU noise model (Kalibr imu yaml)

The camera model is the dataset's Kalibr calibration (TUM-VI: Kannala-Brandt, EuRoC: pinhole
radtan); --calib replaces it, eg with Basalt's Double Sphere / EUCM calibrations of TUM-VI and
EuRoC (the observations do not depend on the model).

Two initializations are available (--init):
  input   the state SLAM Dunk's global BA starts from (loop-closed VIO + its observations), to
          compare VIBA's optimization with SLAM Dunk's global optimizer from the same start
  final   SLAM Dunk's optimized state and final (outlier-filtered) observations, to refine it

Conventions of SLAM Dunk's files (verified on its outputs):
  * timestamps are TUM-VI's minus an offset (recording clock, `extraction.json`), we keep them
  * images were upscaled (`scale` of extraction.json, 1.25 for TUM-VI 512 -> 640) with pixel-center
    convention, we convert back
  * the camera order of `estimate/work` may differ from the recording's (on TUM-VI camera 0 is
    cam1), it is found by matching input.bin's intrinsics to rec/calibration.json
  * all pixels are in OpenCV SIFT's convention (+0.25 px at 640 px), we remove that bias
  * poses are T_world_imu, world is approximately gravity aligned

IMU sample convention: VIBA's preintegration holds each IMU sample constant over the period *preceding*
its timestamp (right for Aria's averaged samples), while TUM-VI/EuRoC samples are point samples (as SLAM
Dunk treats them). By default the IMU timestamps are shifted by half a sample period (+2.5 ms at 200 Hz) so
that each hold interval is centered on the sample, see --imu-stamp-shift-ms.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "viba_input"))
import vibaio  # noqa: E402

SIFT_PIXEL_BIAS = 0.25  # OpenCV SIFT's pixel bias, in SLAM Dunk's (upscaled) pixels
KLT_POINT_WEIGHT = 0.25  # point weight that marks KLT tracks in input.bin.weights


# ------------------------------------------------------------------------------------------------
# SLAM Dunk files


def load_input_bin(path):
    """estimate/work/input.bin: header int64 N P O I, then float64 arrays."""
    n, p, o, i = (int(v) for v in np.fromfile(path, np.int64, 4))
    d = np.fromfile(path, np.float64, offset=32)
    expected = 32 + 14 + 11 * n + 3 * p + 5 * o + 7 * i
    if d.size != expected:
        raise ValueError(f"{path}: unexpected size {d.size} (expected {expected} doubles)")
    sizes = dict(K=32, ext=14, t=n, pose=7 * n, vel=3 * n, pts=3 * p, obs=5 * o, imu=7 * i)
    shapes = dict(K=(2, 16), ext=(2, 7), t=(n,), pose=(n, 7), vel=(n, 3), pts=(p, 3), obs=(o, 5), imu=(i, 7))
    out, off = {}, 0
    for key, size in sizes.items():
        out[key] = d[off : off + size].reshape(shapes[key])
        off += size
    return out


def load_candidates(work):
    m, _ = np.fromfile(work / "input.bin.candidates", np.int64, 2)
    track_to_point = np.fromfile(work / "input.bin.candidates", np.int64, m, offset=16)
    cand = np.fromfile(work / "input.bin.candidates", np.float64, offset=16 + 8 * m).reshape(-1, 8)
    return track_to_point, cand, np.fromfile(work / "input.bin.candidateweights")


def final_observation_weights(work, inp, final_obs):
    """Per-observation weight of ba.txt.observations: matched to input.bin observations, or to the
    candidates that were recovered during the BA."""
    obs_w = np.fromfile(work / "input.bin.obsweights")
    key_in = {tuple(r): j for j, r in enumerate(inp["obs"])}
    track_to_point, cand, cand_w = load_candidates(work)
    key_cand = {}
    for j, r in enumerate(cand):
        pid = track_to_point[int(r[0])]
        if pid >= 0:
            key_cand[(r[1], r[2], float(pid), r[3], r[4])] = j
    w = np.ones(len(final_obs))
    unmatched = 0
    for j, r in enumerate(final_obs):
        k = tuple(r)
        if k in key_in:
            w[j] = obs_w[key_in[k]]
        elif k in key_cand:
            w[j] = cand_w[key_cand[k]]
        else:
            unmatched += 1
    if unmatched:
        print(f"warning: {unmatched} final observations without weight, using 1")
    return w


# ------------------------------------------------------------------------------------------------


def to_source_pixels(uv, scale, remove_sift_bias):
    """SLAM Dunk pixels (images upscaled by `scale`, pixel-center convention) -> source pixels."""
    uv = uv - (SIFT_PIXEL_BIAS if remove_sift_bias else 0.0)
    return (uv + 0.5) / scale - 0.5


def work_camera_order(work_K, rec):
    """Index in rec/calibration.json (= dataset order) of each camera of estimate/work."""
    calib = json.loads((rec / "calibration.json").read_text())
    names = sorted(calib)
    rec_K = [np.array(calib[n]["params"], float) for n in names]
    order = []
    for k in work_K:
        match = [i for i, r in enumerate(rec_K) if r.shape == k.shape and np.allclose(r, k, rtol=0, atol=1e-9)]
        if len(match) != 1:
            raise SystemExit(f"cannot match estimate/work intrinsics to {rec / 'calibration.json'}")
        order.append(int(names[match[0]].removeprefix("cam")))
    return np.array(order)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="SLAM Dunk run, eg results/room1/run2")
    ap.add_argument("--rec", required=True, type=Path, help="SLAM Dunk recording, eg rec/room1")
    ap.add_argument("--dataset", "--tumvi", dest="dataset", required=True, type=Path,
                    help="ASL dataset folder (with mav0/), eg dataset-room1_512_16")
    ap.add_argument("--out", required=True, type=Path, help="output folder (VIBA input)")
    ap.add_argument("--init", choices=["input", "final"], default="input")
    ap.add_argument("--calib", type=Path, default=None,
                    help="camera calibration replacing the dataset's: Kalibr camchain .yaml, Basalt .json "
                         "(eg tumvi_512_ds_calib.json) or a mav0 folder")
    ap.add_argument("--valid-radius", type=float, default=None,
                    help="valid image radius [px] (default: 300 on TUM-VI 512, none otherwise)")
    ap.add_argument("--sigma-px", type=float, default=0.5, help="observation std (source image pixels)")
    ap.add_argument("--klt-weight", type=float, default=None,
                    help="information multiplier of KLT points (default: SLAM Dunk's point weight)")
    ap.add_argument("--keep-sift-bias", action="store_true", help="do not remove the +0.25 px bias")
    ap.add_argument("--imu-stamp-shift-ms", type=float, default=None,
                    help="added to IMU timestamps (default: half the median sample period; 0 disables)")
    args = ap.parse_args()

    work = args.run / "estimate" / "work"
    extraction = json.loads((args.rec / "extraction.json").read_text())
    offset_ns = int(extraction["offset_ns"])
    scale = float(extraction.get("scale", 1.25))
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---- calibration (device frame = IMU frame)
    cams, noise = vibaio.dataset_calibration(args.dataset)
    imu_calib = vibaio.Imu("imu0")
    valid_radius = args.valid_radius if args.valid_radius is not None else vibaio.default_valid_radius(
        args.dataset, cams)
    if args.calib is not None:
        cams, _, _ = vibaio.override_calibration(cams, args.calib)
        # biases / noise of the dataset are kept, eg Basalt's are its own estimates
    for c in cams:
        c.valid_radius = valid_radius
    device_class = "TUM-VI" if vibaio.is_tumvi(args.dataset) else "ASL"
    vibaio.write_imu_noise(out, noise)

    # ---- IMU samples (recording clock)
    t_imu_ns, gyro, accel = vibaio.read_asl_imu(args.dataset / "mav0")
    t_imu_ns = t_imu_ns - offset_ns
    imu_shift_ns = vibaio.imu_stamp_shift_ns(t_imu_ns, args.imu_stamp_shift_ms)
    t_imu_viba_ns = t_imu_ns + imu_shift_ns  # timestamps as written for VIBA
    vibaio.write_imu(out, t_imu_viba_ns, gyro, accel)

    # ---- keyframe states
    inp = load_input_bin(work / "input.bin")
    kf_ns = np.load(work / "export.npz")["t"].astype(np.int64)
    assert len(kf_ns) == len(inp["t"])
    if args.init == "input":
        q_wxyz = inp["pose"][:, :4]
        p = inp["pose"][:, 4:7]
        v = inp["vel"]
        obs = inp["obs"]
        obs_w = np.fromfile(work / "input.bin.obsweights")
    else:
        ba = np.loadtxt(work / "ba.txt")
        assert np.array_equal(ba[:, 0].astype(np.int64), kf_ns)
        q_wxyz = ba[:, [7, 4, 5, 6]]
        p = ba[:, 1:4]
        v = np.loadtxt(work / "ba.txt.vel")[:, 1:4]
        obs = np.fromfile(work / "ba.txt.observations").reshape(-1, 5)
        obs_w = final_observation_weights(work, inp, obs)

    # angular velocity at keyframes: raw gyro (nearest sample), bias is small for TUM-VI
    omega = vibaio.nearest_gyro(t_imu_ns, gyro, kf_ns)

    # keyframes need IMU samples around them (preintegration), drop those outside the IMU range
    margin_ns = 10_000_000
    keep_kf = (kf_ns > t_imu_viba_ns[0] + margin_ns) & (kf_ns < t_imu_viba_ns[-1] - margin_ns)
    if not keep_kf.all():
        print(f"dropping {np.count_nonzero(~keep_kf)} keyframes outside of the IMU time range")

    kf_us = kf_ns // 1000  # VIBA's tracking timestamps are in microseconds
    assert len(np.unique(kf_us)) == len(kf_us)
    kk = np.flatnonzero(keep_kf)
    vibaio.write_trajectory(out, kf_us[kk], kf_ns[kk] + offset_ns, p[kk], q_wxyz[kk], v[kk], omega[kk],
                            session="slamdunk")
    vibaio.write_calibration(out, cams, imu_calib, kf_us[kk], kf_ns[kk] + offset_ns, device_class)

    # ---- observations
    point_w = np.fromfile(work / "input.bin.weights")
    kf = obs[:, 0].astype(int)
    cam = work_camera_order(inp["K"], args.rec)[obs[:, 1].astype(int)]
    pid = obs[:, 2].astype(np.int64)
    uv = to_source_pixels(obs[:, 3:5], scale, not args.keep_sift_bias)
    pw = point_w[pid]
    if args.klt_weight is not None:
        pw = np.where(pw == KLT_POINT_WEIGHT, args.klt_weight, pw)
    info_sqrt = np.sqrt(pw * obs_w) / args.sigma_px
    sel = np.flatnonzero(keep_kf[kf])
    vibaio.write_observations(out, pid[sel], kf_us[kf[sel]], cam[sel], uv[sel], info_sqrt[sel])

    n_klt = int((point_w[pid] == KLT_POINT_WEIGHT).sum())
    print(f"{out}: {np.count_nonzero(keep_kf)} keyframes, {len(np.unique(pid))} points, {len(obs)} observations "
          f"({n_klt} KLT), {len(t_imu_ns)} IMU samples, init={args.init}, clock offset {offset_ns} ns, "
          f"IMU stamp shift {imu_shift_ns / 1e6:.3f} ms, cameras {[c.model for c in cams]}")


if __name__ == "__main__":
    main()
