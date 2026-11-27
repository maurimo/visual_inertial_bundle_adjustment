#!/usr/bin/env python3
"""Convert a SLAM Dunk run on a Project Aria (LaMAria) recording into a VI-BA input folder.

Modelled on VI-BA's tools/slamdunk_to_viba/slamdunk_to_viba.py (which handles ASL datasets), for the
Aria-native SLAM Dunk pipeline (recording extracted with SLAM Dunk's tools/extract_recording.py).

Inputs
  --run   SLAM Dunk run folder (with estimate/work/{ba.txt,ba.txt.vel,ba.txt.observations,input.bin*,export.npz})
  --rec   SLAM Dunk recording folder (calibration.json, imu0/data.csv, only used for consistency checks)
  --vrs   output folder of VI-BA's tools/process_vrs/process_vrs.py on the recording's .vrs:
          factory_calibration.json (as projectaria_tools exports it) and imu_samples_imu-right.csv (raw)
  --out   VI-BA input folder to write

Written files (the form VI-BA expects for Aria)
  factory_calibration.json   the VRS factory calibration reduced to camera-slam-left, camera-slam-right and
                             imu-right; device frame = Aria device frame (= camera-slam-left), unchanged
  online_calibration.jsonl   one entry per keyframe, all equal to the factory calibration (initial guess)
  open_loop_trajectory.csv   keyframe states (T_world_device, velocity and angular velocity of the device)
  session_observations.csv   visual observations at the keyframes, native 640x480 pixels
  imu_samples_imu-right.csv  raw (unrectified) imu-right samples with their VRS capture timestamps
  vrs_source_info.json       cameras [slam-left, slam-right] (serial numbers), IMUs [imu-right]

Conventions (see README.md for how each was verified)
  * SLAM Dunk body frame = imu-right; ba.txt holds T_world_body at the keyframe image timestamps
    (capture timestamp of camera-slam-left, device clock), quaternion order x y z w.
  * recording cam0/cam1 = camera-slam-left/right (VRS streams 1201-1 / 1201-2), but the cameras of
    estimate/work are swapped (camera 0 = slam-right, "stereo-reference-v1"); the order is found by
    matching intrinsics. Observations are in native image pixels (no rescaling), in OpenCV SIFT's
    convention (enable_precise_upscale off):
    detected = true + 0.25 px; SLAM Dunk adds the same 0.25 px to its optical-flow tracks. We
    subtract it, so that pixel (0, 0) is the centre of the top-left pixel (Aria convention).
  * VI-BA reference clock = keyframe (tracking) timestamps; an IMU sample stamped t is held over
    the period preceding t and refers to reference time t - TimeOffsetSec_Device_{Gyro,Accel}.
    Raw samples and timestamps are passed unchanged (as VI-BA's own Aria tooling does).
  * utc_timestamp_ns carries the exact keyframe timestamp in ns (tracking timestamps are in us).

Two initializations (--init): 'final' = SLAM Dunk's optimized keyframes and final (outlier-filtered)
observations; 'input' = the state and observations its global BA starts from.
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
VIBA_TOOLS = HERE.parent
sys.path.insert(0, str(VIBA_TOOLS / "viba_input"))
sys.path.insert(0, str(VIBA_TOOLS / "slamdunk_to_viba"))
import vibaio  # noqa: E402
from slamdunk_to_viba import final_observation_weights, load_input_bin  # noqa: E402

SIFT_PIXEL_BIAS = 0.25
KLT_POINT_WEIGHT = 0.25
CAMERA_LABELS = ["camera-slam-left", "camera-slam-right"]
IMU_LABEL = "imu-right"


def se3_from_json(j):
    q = [j["UnitQuaternion"][0], *j["UnitQuaternion"][1]]
    return vibaio.make_T(vibaio.quat_to_rot(q), j["Translation"])


def quat_xyzw_to_rot(q):
    return vibaio.quat_to_rot([q[3], q[0], q[1], q[2]])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--rec", required=True, type=Path)
    ap.add_argument("--vrs", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--init", choices=["input", "final"], default="final")
    ap.add_argument("--sigma-px", type=float, default=0.5, help="observation std of a weight-1 observation [px]")
    ap.add_argument("--klt-weight", type=float, default=None,
                    help="information multiplier of optical-flow points (default: SLAM Dunk's, 0.25)")
    ap.add_argument("--keep-sift-bias", action="store_true", help="do not remove the +0.25 px bias")
    args = ap.parse_args()

    work = args.run / "estimate" / "work"
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---- factory calibration: slam cameras + imu-right of the VRS calibration, otherwise unchanged
    fac = json.loads((args.vrs / "factory_calibration.json").read_text())
    cams_json = [next(c for c in fac["CameraCalibrations"] if c["Label"] == lab) for lab in CAMERA_LABELS]
    imu_json = next(c for c in fac["ImuCalibrations"] if c["Label"] == IMU_LABEL)
    for c in cams_json:
        assert c["Projection"]["Name"] == "FisheyeRadTanThinPrism" and len(c["Projection"]["Params"]) == 15
        assert c["ConfigData"]["ImageWidth"] == 640 and c["ConfigData"]["ImageHeight"] == 480
    factory = {"DeviceClassInfo": fac["DeviceClassInfo"], "CameraCalibrations": cams_json,
               "ImuCalibrations": [imu_json]}
    (out / "factory_calibration.json").write_text(json.dumps(factory, indent=2))
    (out / "vrs_source_info.json").write_text(json.dumps(
        {"camera_ids": [c["SerialNumber"] for c in cams_json], "imu_ids": [IMU_LABEL]}, indent=2))

    T_dev_imu = se3_from_json(imu_json["T_Device_Imu"])
    T_imu_dev = vibaio.invert(T_dev_imu)
    T_dev_cam = [se3_from_json(c["T_Device_Camera"]) for c in cams_json]

    # consistency with the calibration SLAM Dunk ran with (rec/calibration.json = LaMAria's aria_calibration.json)
    rec_cal = json.loads((args.rec / "calibration.json").read_text())
    for i, c in enumerate(cams_json):
        rc = rec_cal[f"cam{i}"]
        p = np.array(rc["params"])
        d_intr = np.abs(np.r_[p[0], p[2:]] - np.array(c["Projection"]["Params"])).max()
        T_b_s = vibaio.make_T(quat_xyzw_to_rot(rc["T_b_s"]["qvec"]), rc["T_b_s"]["tvec"])
        dT = vibaio.invert(T_b_s) @ T_imu_dev @ T_dev_cam[i]
        d_rot = np.linalg.norm(vibaio.so3_log(dT[:3, :3]))
        d_pos = np.linalg.norm(dT[:3, 3])
        print(f"cam{i} = {c['Label']}: SLAM Dunk calibration vs VRS factory: intrinsics {d_intr:.2e}, "
              f"rotation {d_rot:.2e} rad, translation {d_pos:.2e} m")
        assert d_intr < 1e-9 and d_rot < 1e-9 and d_pos < 1e-9

    # ---- IMU samples: raw, as extracted by process_vrs.py
    imu_file = args.vrs / f"imu_samples_{IMU_LABEL}.csv"
    shutil.copyfile(imu_file, out / f"imu_samples_{IMU_LABEL}.csv")
    imu = np.loadtxt(imu_file, delimiter=",", skiprows=1, usecols=(0, 2, 3, 4, 5, 6, 7))
    t_imu_ns = np.loadtxt(imu_file, delimiter=",", skiprows=1, usecols=0, dtype=np.int64)
    assert np.all(np.diff(t_imu_ns) > 0), "IMU timestamps are not strictly increasing"
    rec_imu_t = np.loadtxt(args.rec / "imu0" / "data.csv", delimiter=",", skiprows=1, usecols=0, dtype=np.int64)
    rec_imu = np.loadtxt(args.rec / "imu0" / "data.csv", delimiter=",", skiprows=1)
    assert np.array_equal(rec_imu_t, t_imu_ns), "IMU samples differ from the ones SLAM Dunk read"
    print(f"IMU: {len(t_imu_ns)} raw samples, same timestamps as SLAM Dunk's imu0, max value difference "
          f"{np.abs(rec_imu[:, 1:7] - imu[:, 1:7]).max():.1e}, median period "
          f"{np.median(np.diff(t_imu_ns)) * 1e-6:.4f} ms, max gap {np.diff(t_imu_ns).max() * 1e-6:.3f} ms")

    # ---- keyframe states (T_world_imu-right)
    inp = load_input_bin(work / "input.bin")
    kf_ns = np.load(work / "export.npz")["t"].astype(np.int64)
    assert len(kf_ns) == len(inp["t"]) and np.abs(inp["t"] * 1e9 - kf_ns).max() < 1000
    # camera order of estimate/work: SLAM Dunk's "stereo-reference-v1" packet swaps the cameras
    # (work camera 0 = recording cam1 = camera-slam-right); found by matching the intrinsics
    work_order = []
    for k in inp["K"]:
        match = [i for i in range(2) if np.allclose(k, rec_cal[f"cam{i}"]["params"], rtol=0, atol=1e-12)]
        assert len(match) == 1, "cannot match estimate/work intrinsics to rec/calibration.json"
        work_order.append(match[0])
    work_order = np.array(work_order)
    assert sorted(work_order) == [0, 1]
    print(f"camera order of estimate/work: {[CAMERA_LABELS[i] for i in work_order]}")
    if args.init == "input":
        q_wxyz, p, v = inp["pose"][:, :4], inp["pose"][:, 4:7], inp["vel"]
        obs = inp["obs"]
        obs_w = np.fromfile(work / "input.bin.obsweights")
    else:
        ba = np.loadtxt(work / "ba.txt")
        assert np.array_equal(ba[:, 0].astype(np.int64), kf_ns)
        q_wxyz, p = ba[:, [7, 4, 5, 6]], ba[:, 1:4]
        v = np.loadtxt(work / "ba.txt.vel")[:, 1:4]
        obs = np.fromfile(work / "ba.txt.observations").reshape(-1, 5)
        obs_w = final_observation_weights(work, inp, obs)
        # SLAM Dunk estimates the gravity vector in its world frame (tilted by a fraction of a degree),
        # VI-BA keeps gravity fixed at (0, 0, -9.81): rotate the world so that gravity is along -z
        g = np.loadtxt(work / "ba.txt.gravity")
        a, b = g / np.linalg.norm(g), np.array([0.0, 0.0, -1.0])
        axis = np.cross(a, b)
        ang = np.arctan2(np.linalg.norm(axis), a @ b)
        R_align = vibaio.so3_exp(axis / max(np.linalg.norm(axis), 1e-300) * ang)
        print(f"SLAM Dunk gravity {g} (|g| = {np.linalg.norm(g):.5f}), world rotated by {np.degrees(ang):.3f} deg")
        p, v = p @ R_align.T, v @ R_align.T
        q_wxyz = np.stack([vibaio.rot_to_quat(R_align @ vibaio.quat_to_rot(q)) for q in q_wxyz])

    # angular velocity of the IMU at the keyframes: factory-rectified gyro, mean over +-5 ms
    gj = imu_json["Gyroscope"]
    gyro = (imu[:, 1:4] - np.array(gj["Bias"]["Offset"])) @ np.linalg.inv(np.array(gj["Model"]["RectificationMatrix"])).T
    t_gyro_ref = t_imu_ns - int(round(gj["TimeOffsetSec_Device_Gyro"] * 1e9))
    lo = np.searchsorted(t_gyro_ref, kf_ns - 5_000_000)
    hi = np.searchsorted(t_gyro_ref, kf_ns + 5_000_000)
    csum = np.vstack([np.zeros(3), np.cumsum(gyro, axis=0)])
    omega_imu = (csum[np.maximum(hi, lo + 1).clip(max=len(gyro))] - csum[lo.clip(max=len(gyro) - 1)]) \
        / np.maximum(hi - lo, 1)[:, None]

    # keyframes need IMU samples around them: VI-BA preintegrates from 5 ms before the first keyframe, and needs
    # a sample stamped after (keyframe time + time offset) at the end (offsets as in the factory calibration,
    # preintegrations are not recomputed with the default settings)
    dts_ns = [int(1e9 * gj["TimeOffsetSec_Device_Gyro"]), int(1e9 * imu_json["Accelerometer"]["TimeOffsetSec_Device_Accel"])]
    margin_ns = 500_000
    keep_kf = (kf_ns - 5_000_000 + min(dts_ns) > t_imu_ns[0] + margin_ns) & (kf_ns + max(dts_ns) < t_imu_ns[-1] - margin_ns)
    if not keep_kf.all():
        print(f"dropping {np.count_nonzero(~keep_kf)} keyframes outside of the IMU time range")

    # device (Aria device frame) states from the IMU states
    N = len(kf_ns)
    R_w_imu = np.stack([vibaio.quat_to_rot(q) for q in q_wxyz])
    R_w_dev = R_w_imu @ T_imu_dev[:3, :3]
    p_dev = p + R_w_imu @ T_imu_dev[:3, 3]
    q_dev = np.stack([vibaio.rot_to_quat(R) for R in R_w_dev])
    omega_dev = omega_imu @ T_imu_dev[:3, :3]  # R_dev_imu omega_imu
    # VI-BA: v_imu = v_dev + R_w_dev (omega_dev x t_dev_imu)
    v_dev = v - np.einsum("nij,nj->ni", R_w_dev, np.cross(omega_dev, T_dev_imu[:3, 3]))

    kf_us = kf_ns // 1000
    assert len(np.unique(kf_us)) == N
    kk = np.flatnonzero(keep_kf)
    vibaio.write_trajectory(out, kf_us[kk], kf_ns[kk], p_dev[kk], q_dev[kk], v_dev[kk], omega_dev[kk],
                            session="slamdunk")
    with open(out / "online_calibration.jsonl", "w") as f:
        for k in kk:
            f.write(json.dumps({"tracking_timestamp_us": int(kf_us[k]), "utc_timestamp_ns": int(kf_ns[k]),
                                "CameraCalibrations": cams_json, "ImageSizes": [[640, 480], [640, 480]],
                                "ImuCalibrations": [imu_json]}) + "\n")

    # ---- observations
    point_w = np.fromfile(work / "input.bin.weights")
    kf = obs[:, 0].astype(int)
    cam = work_order[obs[:, 1].astype(int)]
    pid = obs[:, 2].astype(np.int64)
    uv = obs[:, 3:5] - (0.0 if args.keep_sift_bias else SIFT_PIXEL_BIAS)
    pw = point_w[pid]
    if args.klt_weight is not None:
        pw = np.where(pw == KLT_POINT_WEIGHT, args.klt_weight, pw)
    info_sqrt = np.sqrt(pw * obs_w) / args.sigma_px
    sel = np.flatnonzero(keep_kf[kf])
    vibaio.write_observations(out, pid[sel], kf_us[kf[sel]], cam[sel], uv[sel], info_sqrt[sel])

    n_klt = int((point_w[pid] == KLT_POINT_WEIGHT).sum())
    dur = (kf_ns[kk[-1]] - kf_ns[kk[0]]) * 1e-9
    print(f"{out}: {len(kk)} keyframes over {dur:.1f} s (median spacing {np.median(np.diff(kf_ns)) * 1e-9:.2f} s), "
          f"{len(np.unique(pid))} points, {len(obs)} observations ({n_klt} optical flow), init={args.init}, "
          f"sqrt-info range {info_sqrt.min():.2f}..{info_sqrt.max():.2f} (median {np.median(info_sqrt):.2f})")


if __name__ == "__main__":
    main()
