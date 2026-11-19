#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Export a VIBA input folder (keyframe poses, cameras, observations) as a COLMAP text model, eg to
inspect it in COLMAP's GUI, or as a test of colmap_to_viba.py (--sim3 applies a similarity, as
if the model came from structure from motion).

Points are triangulated from the observations and the keyframe poses. Image names are
`<camera label>/data/<utc timestamp ns>.png` (ASL layout). Double Sphere / EUCM / UCM cameras have
no COLMAP equivalent and are refused.

  viba_to_colmap.py <viba input dir> <colmap out dir> [--sim3 s,rx,ry,rz,tx,ty,tz]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_viba_input as chk  # noqa: E402
import colmap_io  # noqa: E402
import vibaio  # noqa: E402


def to_colmap_camera(cam):
    p = list(cam.params)
    if cam.model == "Linear":
        model, q = "PINHOLE", p[:4]
    elif cam.model == "PinholeRadTan":
        model, q = "OPENCV", p[:8]
    elif cam.model == "PinholeRadTanK3":
        model, q = "FULL_OPENCV", p[:9] + [0.0, 0.0, 0.0]
    elif cam.model == "KannalaBrandtK3":
        model, q = "OPENCV_FISHEYE", p[:8]
    elif cam.model == "FisheyeRadTanThinPrism":
        model, q = "RAD_TAN_THIN_PRISM_FISHEYE", [p[0], p[0]] + p[1:15]
    else:
        raise SystemExit(f"{cam.label}: no COLMAP camera model for {cam.model}")
    q[2] += 0.5
    q[3] += 0.5
    return model, q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--sim3", default=None, help="s,rx,ry,rz,tx,ty,tz: X_model = s R(r) X_world + t")
    ap.add_argument("--min-obs", type=int, default=2)
    args = ap.parse_args()

    cams = chk.load_cameras(args.input / "factory_calibration.json")
    tr = np.genfromtxt(args.input / "open_loop_trajectory.csv", delimiter=",", names=True, dtype=None, encoding=None)
    t_us = np.array([int(v) for v in tr["tracking_timestamp_us"]], dtype=np.int64)
    utc = {int(a): int(b) for a, b in zip(tr["tracking_timestamp_us"], tr["utc_timestamp_ns"])}
    T_w_i = {}
    for k in range(len(t_us)):
        q = [tr[f"q{a}_odometry_device"][k] for a in "wxyz"]
        T_w_i[t_us[k]] = vibaio.make_T(vibaio.quat_to_rot(q), [tr[f"t{a}_odometry_device"][k] for a in "xyz"])

    s, R, tr3 = 1.0, np.eye(3), np.zeros(3)
    if args.sim3:
        v = [float(x) for x in args.sim3.split(",")]
        s, R, tr3 = v[0], vibaio.so3_exp(np.array(v[1:4])), np.array(v[4:7])

    obs = np.loadtxt(args.input / "session_observations.csv", delimiter=",", skiprows=1)
    pid, ot, oc, ouv = obs[:, 0].astype(np.int64), obs[:, 1].astype(np.int64), obs[:, 2].astype(int), obs[:, 3:5]
    keep = np.isin(ot, t_us)
    pid, ot, oc, ouv = pid[keep], ot[keep], oc[keep], ouv[keep]

    ccams = {}
    for i, c in enumerate(cams):
        model, q = to_colmap_camera(c)
        ccams[i + 1] = colmap_io.ColmapCamera(i + 1, model, c.width, c.height, np.array(q))

    # images: one per (keyframe, camera) with observations
    image_id, images = {}, {}
    for ts, c in sorted(set(zip(ot.tolist(), oc.tolist()))):
        T_w_c = T_w_i[ts] @ cams[c].T_imu_cam
        R_m_c = R @ T_w_c[:3, :3]
        c_m = s * R @ T_w_c[:3, 3] + tr3
        R_c_m = R_m_c.T
        iid = len(images) + 1
        image_id[(ts, c)] = iid
        images[iid] = colmap_io.ColmapImage(iid, vibaio.rot_to_quat(R_c_m), -R_c_m @ c_m, c + 1,
                                            f"{cams[c].label}/data/{utc[ts]}.png", [], [])

    rays_all = np.ones((len(pid), 3))
    for c in np.unique(oc):
        rays_all[oc == c, :2] = chk.unproject(cams[c], ouv[oc == c])

    points = {}
    order = np.argsort(pid, kind="stable")
    starts = np.r_[0, np.flatnonzero(np.diff(pid[order])) + 1, len(pid)]
    n_bad = 0
    for a, b in zip(starts[:-1], starts[1:]):
        sel = order[a:b]
        if len(np.unique(ot[sel])) < args.min_obs:
            continue
        T_w_c = [T_w_i[tt] @ cams[cc].T_imu_cam for tt, cc in zip(ot[sel], oc[sel])]
        X = chk.refine(cams, T_w_c, oc[sel], ouv[sel], chk.triangulate(T_w_c, rays_all[sel]))
        err = chk.reprojection(cams, T_w_c, oc[sel], ouv[sel], X)
        if np.isnan(err).any():
            n_bad += 1
            continue
        p_id = int(pid[sel[0]])
        track_img, track_idx = [], []
        for tt, cc, z in zip(ot[sel], oc[sel], ouv[sel]):
            im = images[image_id[(tt, cc)]]
            track_img.append(im.id)
            track_idx.append(len(im.xys))
            im.xys.append(z + 0.5)
            im.point3D_ids.append(p_id)
        points[p_id] = colmap_io.ColmapPoint(p_id, s * R @ X + tr3, np.array([128, 128, 128]), float(np.mean(err)),
                                             np.array(track_img), np.array(track_idx))
    for im in images.values():
        im.xys = np.array(im.xys).reshape(-1, 2)
        im.point3D_ids = np.array(im.point3D_ids, np.int64)
    colmap_io.write_text(args.out, ccams, images, points)
    print(f"{args.out}: {len(ccams)} cameras, {len(images)} images, {len(points)} points "
          f"({n_bad} dropped behind a camera)")


if __name__ == "__main__":
    main()
