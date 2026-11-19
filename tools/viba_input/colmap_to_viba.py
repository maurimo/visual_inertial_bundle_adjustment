#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Convert a COLMAP sparse model of a visual-inertial recording (ASL layout) into a VIBA input folder.

COLMAP models (text or binary; COLMAP, GLOMAP, hloc, LaMAria's pipeline, ...) hold camera poses
of arbitrary scale and orientation, the intrinsics, and the 2d tracks of the 3d points. VIBA
also needs the IMU samples, the camera-IMU extrinsics and a metric, gravity-aligned initial
trajectory, so:

  * image names give camera and timestamp: by default `.../<cam label>/.../<t ns>.<ext>`, eg
    `cam0/data/1520530308199447626.png` (ASL), see --name-regex
  * the IMU, its noise and the camera-IMU extrinsics come from the ASL dataset (--dataset) or
    --calib; the intrinsics from COLMAP (default) or from the calibration (--intrinsics calib)
  * scale and gravity (--align):
      imu    visual-inertial alignment: scale, gravity and velocities by linear least squares on
             IMU preintegrations between consecutive keyframes (biases neglected)
      traj   Sim(3) alignment of the keyframe positions to a reference trajectory (--align-traj,
             formats of vibaio.read_trajectory), eg a VIO output
      none   the model is already metric and gravity aligned (z up)

COLMAP's pixel (0.5, 0.5) is the center of the top-left pixel; pixels and principal points are
shifted by -0.5 for VIBA.

Example:
  colmap_to_viba.py --model sparse/0 --dataset /galway/data/tumvi/dataset-room1_512_16 --out /tmp/viba_in
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import colmap_io  # noqa: E402
import vibaio  # noqa: E402

G = 9.81


def preintegrate(t_imu, gyro, accel, t0, t1):
    """alpha, beta (in the IMU frame at t0), dR, dt, with a zero-order hold of each sample over the
    interval centered on it (samples are point samples) and no bias."""
    i0 = max(np.searchsorted(t_imu, t0) - 1, 0)
    i1 = min(np.searchsorted(t_imu, t1) + 1, len(t_imu))
    ts = t_imu[i0:i1]
    edges = np.r_[t0, np.clip(0.5 * (ts[1:] + ts[:-1]), t0, t1), t1]  # sample k holds on [edges_k, edges_k+1]
    alpha, beta, dR = np.zeros(3), np.zeros(3), np.eye(3)
    for k in range(len(ts)):
        dt = (edges[k + 1] - edges[k]) * 1e-9
        if dt <= 0:
            continue
        a = dR @ accel[i0 + k]
        alpha += beta * dt + 0.5 * a * dt * dt
        beta += a * dt
        dR = dR @ vibaio.so3_exp(gyro[i0 + k] * dt)
    return alpha, beta, dR, (t1 - t0) * 1e-9


def vi_align(t_kf, c, R_wi, d, t_imu, gyro, accel, max_dt):
    """Scale s, gravity g (world), velocities v: p_k = s c_k + d_k, for consecutive keyframes
      p_j = p_i + v_i dt + g dt^2 / 2 + R_i alpha,  v_j = v_i + g dt + R_i beta."""
    n = len(t_kf)
    pairs = [(i, i + 1) for i in range(n - 1) if (t_kf[i + 1] - t_kf[i]) * 1e-9 <= max_dt]
    if len(pairs) < 10:
        raise SystemExit("too few consecutive keyframes for the visual-inertial alignment")
    pre = {p: preintegrate(t_imu, gyro, accel, t_kf[p[0]], t_kf[p[1]]) for p in pairs}

    def solve(g_fixed):
        nx = 3 * n + (0 if g_fixed is not None else 3) + 1
        rows, rhs = [], []
        for (i, j) in pairs:
            alpha, beta, _, dt = pre[(i, j)]
            w = 1.0 / dt  # equal weight per second of data
            A = np.zeros((6, nx))
            b = np.zeros(6)
            A[0:3, 3 * i:3 * i + 3] = -dt * np.eye(3)
            A[0:3, -1] = c[j] - c[i]
            b[0:3] = R_wi[i] @ alpha - (d[j] - d[i])
            A[3:6, 3 * j:3 * j + 3] = np.eye(3)
            A[3:6, 3 * i:3 * i + 3] -= np.eye(3)
            b[3:6] = R_wi[i] @ beta
            if g_fixed is None:
                A[0:3, 3 * n:3 * n + 3] = -0.5 * dt * dt * np.eye(3)
                A[3:6, 3 * n:3 * n + 3] = -dt * np.eye(3)
            else:
                b[0:3] += 0.5 * dt * dt * g_fixed
                b[3:6] += dt * g_fixed
            rows.append(w * A)
            rhs.append(w * b)
        x = np.linalg.lstsq(np.vstack(rows), np.concatenate(rhs), rcond=None)[0]
        return x

    x = solve(None)
    g = x[3 * n:3 * n + 3]
    print(f"VI alignment: unconstrained |g| = {np.linalg.norm(g):.3f} m/s^2, scale {x[-1]:.6g}")
    for _ in range(3):  # fix |g| = G, refine its direction
        g = G * g / np.linalg.norm(g)
        x = solve(g)
        s, v = x[-1], x[:3 * n].reshape(n, 3)
        # update the gravity direction from the residual of the velocity equations
        res = []
        for (i, j) in pairs:
            _, beta, _, dt = pre[(i, j)]
            res.append((v[j] - v[i] - R_wi[i] @ beta) / dt)
        g_new = np.mean(res, axis=0)
        if np.degrees(np.arccos(np.clip(g_new @ g / (np.linalg.norm(g_new) * G), -1, 1))) < 1e-3:
            break
        g = g_new
    if s <= 0:
        raise SystemExit(f"visual-inertial alignment failed (scale {s})")
    return s, G * g / np.linalg.norm(g), v


def vi_velocities(t_kf, p, R_wi, t_imu, gyro, accel, g, max_dt):
    """Velocities by least squares on the preintegration equations, positions and gravity known.
    Keyframes without a neighbor within max_dt get central differences of the positions."""
    n = len(t_kf)
    rows, rhs = [], []
    for i in range(n - 1):
        dt = (t_kf[i + 1] - t_kf[i]) * 1e-9
        if dt > max_dt:
            continue
        alpha, beta, _, dt = preintegrate(t_imu, gyro, accel, t_kf[i], t_kf[i + 1])
        w = 1.0 / dt
        A = np.zeros((6, 3 * n))
        A[0:3, 3 * i:3 * i + 3] = dt * np.eye(3)  # p_j - p_i - g dt^2/2 - R alpha = v_i dt
        A[3:6, 3 * i + 3:3 * i + 6] = np.eye(3)  # v_j - v_i = g dt + R beta
        A[3:6, 3 * i:3 * i + 3] = -np.eye(3)
        b = np.r_[p[i + 1] - p[i] - 0.5 * g * dt * dt - R_wi[i] @ alpha, g * dt + R_wi[i] @ beta]
        rows.append(w * A)
        rhs.append(w * b)
    v = np.gradient(p, t_kf * 1e-9, axis=0)
    if rows:
        # weak prior on the finite differences, for keyframes the IMU equations do not reach
        A = np.vstack(rows + [1e-3 * np.eye(3 * n)])
        b = np.concatenate(rhs + [1e-3 * v.ravel()])
        v = np.linalg.lstsq(A, b, rcond=None)[0].reshape(n, 3)
    return v


def umeyama(src, dst):
    """Sim(3) (s, R, t) minimizing |dst - (s R src + t)|."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    a, b = src - mu_s, dst - mu_d
    U, S, Vt = np.linalg.svd(b.T @ a / len(src))
    D = np.eye(3)
    D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (a**2).sum(1).mean()
    return s, R, mu_d - s * R @ mu_s


def rotation_aligning(a, b):
    """Rotation R with R a/|a| = b/|b|."""
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v, cth = np.cross(a, b), a @ b
    if np.linalg.norm(v) < 1e-12:
        return np.eye(3) if cth > 0 else np.diag([1.0, -1.0, -1.0])
    return vibaio.so3_exp(v / np.linalg.norm(v) * np.arctan2(np.linalg.norm(v), cth))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path, help="COLMAP model folder (images.bin or images.txt)")
    ap.add_argument("--dataset", required=True, type=Path, help="ASL dataset folder (IMU, calibration)")
    ap.add_argument("--out", required=True, type=Path, help="output folder (VIBA input)")
    ap.add_argument("--calib", type=Path, default=None,
                    help="calibration replacing the dataset's (Kalibr .yaml, Basalt .json): extrinsics, and "
                         "intrinsics with --intrinsics calib")
    ap.add_argument("--intrinsics", choices=["colmap", "calib"], default="colmap")
    ap.add_argument("--name-regex", default=r"(?P<cam>cam\d+)/(?:.*/)?(?P<t>\d+)\.\w+$",
                    help="regex on image names with groups `cam` (calibration label) and `t` (timestamp ns)")
    ap.add_argument("--align", choices=["imu", "traj", "none"], default="imu")
    ap.add_argument("--align-traj", type=Path, default=None, help="reference trajectory for --align traj")
    ap.add_argument("--max-kf-gap", type=float, default=1.0,
                    help="max time between keyframes used by the IMU alignment [s]")
    ap.add_argument("--min-track-kf", type=int, default=2)
    ap.add_argument("--sigma-px", type=float, default=0.5, help="observation std [px]")
    ap.add_argument("--valid-radius", type=float, default=None)
    ap.add_argument("--imu-stamp-shift-ms", type=float, default=None,
                    help="added to IMU timestamps (default: half the median sample period; 0 disables)")
    args = ap.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    mav0 = args.dataset / "mav0"

    ccams, images, points = colmap_io.read_model(args.model)

    # ---- calibration
    cams, noise = vibaio.dataset_calibration(args.dataset)
    if args.calib is not None:
        cams, _, _ = vibaio.override_calibration(cams, args.calib)
    by_label = {c.label: c for c in cams}
    rx = re.compile(args.name_regex)
    img_info = {}
    for im in images.values():
        m = rx.search(im.name)
        if not m:
            raise SystemExit(f"image name {im.name} does not match --name-regex")
        if m["cam"] not in by_label:
            raise SystemExit(f"camera {m['cam']} of {im.name} is not in the calibration ({list(by_label)})")
        img_info[im.id] = (m["cam"], int(m["t"]))
    if args.intrinsics == "colmap":
        for label, cam in by_label.items():
            cids = sorted({images[i].camera_id for i, (lb, _) in img_info.items() if lb == label})
            if not cids:
                continue
            if len(cids) > 1:
                print(f"warning: {label} has {len(cids)} COLMAP cameras, using camera {cids[0]}")
            cc = ccams[cids[0]]
            conv = vibaio.colmap_camera(label, cc.model, cc.params, cc.width, cc.height)
            cam.model, cam.params, cam.width, cam.height = conv.model, conv.params, conv.width, conv.height
    for c in cams:
        if args.valid_radius is not None:
            c.valid_radius = args.valid_radius
    cam_index = {c.label: i for i, c in enumerate(cams)}

    # ---- IMU
    t_imu, gyro, accel = vibaio.read_asl_imu(mav0)
    offset_ns = int(t_imu[0] // 10**9 - 1) * 10**9
    shift_ns = vibaio.imu_stamp_shift_ns(t_imu, args.imu_stamp_shift_ms)
    t_imu_viba = t_imu - offset_ns + shift_ns
    vibaio.write_imu(out, t_imu_viba, gyro, accel)
    vibaio.write_imu_noise(out, noise)

    # ---- keyframes: one per timestamp, IMU pose from the first camera (in calibration order)
    margin_ns = 10_000_000
    by_time = {}
    for iid, (label, t) in img_info.items():
        if t - offset_ns > t_imu_viba[0] + margin_ns and t - offset_ns < t_imu_viba[-1] - margin_ns:
            by_time.setdefault(t, []).append(iid)
    t_kf = np.array(sorted(by_time), dtype=np.int64)
    c_kf, R_kf, d_kf = [], [], []
    for t in t_kf:
        iid = min(by_time[t], key=lambda i: cam_index[img_info[i][0]])
        im = images[iid]
        R_cw = vibaio.quat_to_rot(im.qvec)
        R_wc = R_cw.T
        center = -R_wc @ im.tvec
        T_cam_imu = vibaio.invert(by_label[img_info[iid][0]].T_imu_cam)
        c_kf.append(center)
        R_kf.append(R_wc @ T_cam_imu[:3, :3])
        d_kf.append(R_wc @ T_cam_imu[:3, 3])  # metric lever arm (camera -> IMU) in world
    c_kf, R_kf, d_kf = np.array(c_kf), np.array(R_kf), np.array(d_kf)

    # ---- scale and gravity
    v_kf = None
    if args.align == "imu":
        s, g, v_kf = vi_align(t_kf, c_kf, R_kf, d_kf, t_imu, gyro, accel, args.max_kf_gap)
        R_align = rotation_aligning(g, np.array([0.0, 0.0, -1.0]))
        p_kf = (s * c_kf + d_kf) @ R_align.T
        v_kf = v_kf @ R_align.T
        R_kf = np.einsum("ij,njk->nik", R_align, R_kf)
        print(f"VI alignment: scale {s:.6g}, gravity {g} (world of the model)")
    elif args.align == "traj":
        if args.align_traj is None:
            raise SystemExit("--align traj needs --align-traj")
        t_tr, p_tr, q_tr, _ = vibaio.read_trajectory(args.align_traj)
        p_ref, _, ok = vibaio.interpolate_poses(t_tr, p_tr, q_tr, t_kf)
        # iterate: positions depend on the scale through the lever arm d_k
        s, R, t = umeyama(c_kf[ok], p_ref[ok])
        for _ in range(3):
            s, R, t = umeyama(c_kf[ok] + d_kf[ok] / s, p_ref[ok])
        p_kf = s * (c_kf @ R.T) + d_kf @ R.T + t
        R_kf = np.einsum("ij,njk->nik", R, R_kf)
        err = np.linalg.norm(p_kf[ok] - p_ref[ok], axis=1)
        print(f"Sim(3) alignment to {args.align_traj}: scale {s:.6g}, RMSE {np.sqrt((err**2).mean()):.4f} m")
    else:
        p_kf = c_kf + d_kf
    q_kf = np.array([vibaio.rot_to_quat(R) for R in R_kf])
    if v_kf is None:
        v_kf = vi_velocities(t_kf, p_kf, R_kf, t_imu, gyro, accel, np.array([0.0, 0.0, -G]), args.max_kf_gap)
    omega = vibaio.nearest_gyro(t_imu, gyro, t_kf)
    kf_us = (t_kf - offset_ns) // 1000
    vibaio.write_trajectory(out, kf_us, t_kf, p_kf, q_kf, v_kf, omega, session="colmap")
    vibaio.write_calibration(out, cams, vibaio.Imu("imu0"), kf_us, t_kf, "COLMAP")

    # ---- observations: the 2d tracks of the 3d points
    kf_of_time = {t: k for k, t in enumerate(t_kf)}
    pid, kfi, cam, uv = [], [], [], []
    for p in points.values():
        for iid, idx in zip(p.image_ids, p.point2D_idxs):
            label, t = img_info[int(iid)]
            if t not in kf_of_time:
                continue
            pid.append(p.id)
            kfi.append(kf_of_time[t])
            cam.append(cam_index[label])
            uv.append(images[int(iid)].xys[idx] - 0.5)
    pid, kfi, cam, uv = np.array(pid, np.int64), np.array(kfi), np.array(cam), np.array(uv).reshape(-1, 2)
    good = vibaio.filter_tracks(pid, kfi, args.min_track_kf)
    vibaio.write_observations(out, pid[good], kf_us[kfi[good]], cam[good], uv[good], 1.0 / args.sigma_px)
    print(f"{out}: {len(t_kf)} keyframes from {len(images)} images, {len(np.unique(pid[good]))} points, "
          f"{good.sum()} observations, cameras {[c.model for c in cams]}, IMU stamp shift {shift_ns / 1e6:.3f} ms")


if __name__ == "__main__":
    main()
