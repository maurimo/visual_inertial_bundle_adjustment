#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Consistency check of a VIBA input folder, without running VIBA.

Triangulates a sample of points from the observations, the keyframe poses of
open_loop_trajectory.csv and the cameras of factory_calibration.json, and prints the
reprojection errors per camera. Wrong extrinsics, pixel conventions, camera order or timestamps
show up as errors of many pixels; a good input gives errors near the observation noise (plus the
drift of the initial trajectory over the length of the tracks).

  check_viba_input.py <viba input dir> [--points 500]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vibaio  # noqa: E402


def load_cameras(path):
    cams = []
    for c in json.loads(Path(path).read_text())["CameraCalibrations"]:
        T = c["T_Device_Camera"]
        q = [T["UnitQuaternion"][0], *T["UnitQuaternion"][1]]
        cams.append(vibaio.Camera(c["Label"], c["Projection"]["Name"], c["Projection"]["Params"],
                                  vibaio.make_T(vibaio.quat_to_rot(q), T["Translation"]),
                                  c["ConfigData"]["ImageWidth"], c["ConfigData"]["ImageHeight"]))
    return cams


def unproject(cam, uv, iters=20):
    """Normalized (x, y) with project(x, y, 1) = uv, Gauss-Newton with numeric Jacobian."""
    f = np.array(cam.params[:2]) if cam.model != "FisheyeRadTanThinPrism" else np.array([cam.params[0]] * 2)
    c = np.array(cam.params[2:4]) if cam.model != "FisheyeRadTanThinPrism" else np.array(cam.params[1:3])
    xy = (uv - c) / f
    h = 1e-6
    for _ in range(iters):
        P = np.c_[xy, np.ones(len(xy))]
        r = cam.project(P) - uv
        jx = (cam.project(P + [h, 0, 0]) - uv - r) / h
        jy = (cam.project(P + [0, h, 0]) - uv - r) / h
        det = jx[:, 0] * jy[:, 1] - jy[:, 0] * jx[:, 1]
        dx = (jy[:, 1] * r[:, 0] - jy[:, 0] * r[:, 1]) / det
        dy = (-jx[:, 1] * r[:, 0] + jx[:, 0] * r[:, 1]) / det
        xy = xy - np.stack([dx, dy], -1)
    return xy


def triangulate(T_w_c, rays_c):
    """Least squares point closest to all rays (world frame)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for T, r in zip(T_w_c, rays_c):
        d = T[:3, :3] @ (r / np.linalg.norm(r))
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ T[:3, 3]
    return np.linalg.lstsq(A, b, rcond=None)[0]


def _project_all(cams, T_c_w, cam_idx, X):
    """Pixels of X in each observation's camera (NaN behind the camera)."""
    Pc = np.einsum("nij,j->ni", T_c_w[:, :3, :3], X) + T_c_w[:, :3, 3]
    uv = np.full((len(Pc), 2), np.nan)
    for c in np.unique(cam_idx):
        m = (cam_idx == c) & (Pc[:, 2] > 1e-6)
        if m.any():
            uv[m] = cams[c].project(Pc[m])
    return uv


def reprojection(cams, T_w_c, cam_idx, uv, X):
    T_c_w = np.linalg.inv(np.asarray(T_w_c))
    return np.linalg.norm(_project_all(cams, T_c_w, np.asarray(cam_idx), X) - uv, axis=1)


def refine(cams, T_w_c, cam_idx, uv, X, iters=5):
    """Gauss-Newton on the reprojection error (numeric Jacobian)."""
    T_c_w = np.linalg.inv(np.asarray(T_w_c))
    cam_idx = np.asarray(cam_idx)

    def res(X):
        return (_project_all(cams, T_c_w, cam_idx, X) - uv).ravel()

    for _ in range(iters):
        r = res(X)
        if not np.isfinite(r).all():
            break
        J = np.stack([(res(X + e) - r) / 1e-6 for e in np.eye(3) * 1e-6], -1)
        X = X - np.linalg.lstsq(J, r, rcond=None)[0]
    return X


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("--points", type=int, default=500, help="number of sampled points")
    ap.add_argument("--min-obs", type=int, default=3)
    ap.add_argument("--max-span-kf", type=int, default=0,
                    help="use only observations within this many keyframes of the first one (0: all), "
                         "limits the effect of trajectory drift")
    args = ap.parse_args()

    cams = load_cameras(args.input / "factory_calibration.json")
    tr = np.genfromtxt(args.input / "open_loop_trajectory.csv", delimiter=",", names=True, dtype=None, encoding=None)
    t_us = np.array([int(v) for v in tr["tracking_timestamp_us"]], dtype=np.int64)
    pose = {}
    for k in range(len(t_us)):
        q = [tr["qw_odometry_device"][k], tr["qx_odometry_device"][k], tr["qy_odometry_device"][k],
             tr["qz_odometry_device"][k]]
        p = [tr[f"t{a}_odometry_device"][k] for a in "xyz"]
        pose[t_us[k]] = vibaio.make_T(vibaio.quat_to_rot(q), p)
    kf_index = {t: i for i, t in enumerate(sorted(pose))}

    obs = np.loadtxt(args.input / "session_observations.csv", delimiter=",", skiprows=1)
    pid, ot, oc, ouv = obs[:, 0].astype(np.int64), obs[:, 1].astype(np.int64), obs[:, 2].astype(int), obs[:, 3:5]
    missing = ~np.isin(ot, t_us)
    if missing.any():
        print(f"WARNING: {missing.sum()} observations at timestamps without a keyframe pose")
    order = np.argsort(pid, kind="stable")
    pid, ot, oc, ouv = pid[order], ot[order], oc[order], ouv[order]
    starts = np.r_[0, np.flatnonzero(np.diff(pid)) + 1, len(pid)]
    groups = [(starts[i], starts[i + 1]) for i in range(len(starts) - 1) if starts[i + 1] - starts[i] >= args.min_obs]
    rng = np.random.default_rng(0)
    sample = rng.choice(len(groups), size=min(args.points, len(groups)), replace=False)

    errs = {i: [] for i in range(len(cams))}
    behind = 0
    for g in sample:
        a, b = groups[g]
        sel = np.arange(a, b)
        sel = sel[np.isin(ot[sel], t_us)]
        if args.max_span_kf > 0 and len(sel):
            k0 = min(kf_index[t] for t in ot[sel])
            sel = sel[np.array([kf_index[t] - k0 <= args.max_span_kf for t in ot[sel]])]
        if len(sel) < 2:
            continue
        T_w_c = [pose[t] @ cams[c].T_imu_cam for t, c in zip(ot[sel], oc[sel])]
        rays = []
        for c, z in zip(oc[sel], ouv[sel]):
            xy = unproject(cams[c], z[None])[0]
            rays.append(np.r_[xy, 1.0])
        X = triangulate(T_w_c, rays)
        X = refine(cams, T_w_c, oc[sel], ouv[sel], X)
        e = reprojection(cams, T_w_c, oc[sel], ouv[sel], X)
        if np.isnan(e).any():
            behind += 1
            continue
        for c, v in zip(oc[sel], e):
            errs[c].append(v)

    print(f"{args.input}: {len(cams)} cameras {[c.model for c in cams]}, {len(t_us)} keyframes, "
          f"{len(np.unique(pid))} points, {len(pid)} observations; {len(sample)} points sampled, "
          f"{behind} behind a camera after triangulation")
    for c, e in errs.items():
        e = np.array(e)
        if len(e):
            print(f"  {cams[c].label}: {len(e)} obs, reprojection error median {np.median(e):.3f} px, "
                  f"p90 {np.percentile(e, 90):.3f} px, > 3 px {100 * np.mean(e > 3):.1f} %")


if __name__ == "__main__":
    main()
