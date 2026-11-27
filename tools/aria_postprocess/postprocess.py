#!/usr/bin/env python3
"""Post-process a VI-BA run on an Aria recording (input written by aria_slamdunk_to_viba.py).

  postprocess.py <viba input dir> <viba output dir> <cam0/data.csv with the frame timestamps> <dest dir>

Writes into <dest dir>:
  traj_kf.txt          optimized keyframes, 't[s] x y z qx qy qz qw', T_world_imu-right
  traj.txt             the same trajectory at every camera frame between the first and last keyframe:
                       the IMU (compensated with VI-BA's calibration of that interval) is integrated from
                       each keyframe to the next, and the small end-point mismatch is spread linearly
  calib_vs_time.csv    one row per calibration window (see the header comment written in the file)
  densify_check.txt    end-point mismatch of the IMU integration (a check of the IMU conventions)
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "viba_input"))
import vibaio  # noqa: E402

GRAVITY = np.array([0.0, 0.0, -9.81])  # VI-BA's fixed gravity
# VI-BA default random walks of the IMU biases (lib/motion/imu_types/ImuNoiseModelParameters.h)
ACC_BIAS_RW_VAR = 1e-8  # m^2/s^4 per second
GYR_BIAS_RW_VAR = 1e-10  # rad^2/s^2 per second
IMU_LABEL = "imu-right"
CAM_LABELS = ["camera-slam-left", "camera-slam-right"]
CAM_SHORT = ["cl", "cr"]
INTR_NAMES = ["f", "cu", "cv", "k0", "k1", "k2", "k3", "k4", "k5", "p0", "p1", "s0", "s1", "s2", "s3"]


def T_json(j):
    return vibaio.make_T(vibaio.quat_to_rot([j["UnitQuaternion"][0], *j["UnitQuaternion"][1]]), j["Translation"])


def stamp(t_ns):
    return f"{t_ns // 10**9}.{t_ns % 10**9:09d}"


def write_traj(path, t_ns, p, R):
    with open(path, "w") as f:
        for t, pp, RR in zip(t_ns, p, R):
            q = vibaio.rot_to_quat(RR)
            f.write(f"{stamp(int(t))} {pp[0]:.9f} {pp[1]:.9f} {pp[2]:.9f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f} {q[0]:.10f}\n")


def imu_params(j):
    """VI-BA's IMU model from its output json: true = S^-1 raw - b, sample stamped t refers to t - dt."""
    a, g = j["Accelerometer"], j["Gyroscope"]
    return dict(Sa=np.array(a["Model"]["RectificationMatrix"]), ba=np.array(a["Bias"]["Offset"]),
                Sg=np.array(g["Model"]["RectificationMatrix"]), bg=np.array(g["Bias"]["Offset"]),
                dta=float(a["TimeOffsetSec_Device_Accel"]), dtg=float(g["TimeOffsetSec_Device_Gyro"]))


def main():
    inp, outd, frames_csv, dest = map(Path, sys.argv[1:5])
    dest.mkdir(parents=True, exist_ok=True)

    fac = json.loads((inp / "factory_calibration.json").read_text())
    fac_imu = next(c for c in fac["ImuCalibrations"] if c["Label"] == IMU_LABEL)
    fac_cams = [next(c for c in fac["CameraCalibrations"] if c["Label"] == lab) for lab in CAM_LABELS]
    T_dev_imu = T_json(fac_imu["T_Device_Imu"])
    T_imu_dev = vibaio.invert(T_dev_imu)
    T_imu_cam_fac = [T_imu_dev @ T_json(c["T_Device_Camera"]) for c in fac_cams]
    fp = imu_params(fac_imu)  # NB: the factory json is in Aria's convention raw = M rect + b
    Ma_inv, Mg_inv = np.linalg.inv(fp["Sa"]), np.linalg.inv(fp["Sg"])

    # ---- optimized keyframes
    d = np.genfromtxt(outd / "open_loop_framerate_trajectory.csv", delimiter=",", names=True, dtype=None, encoding=None)
    kf_ns = np.array([int(v) for v in d["utc_timestamp_ns"]], dtype=np.int64)
    kf_us = np.array([int(v) for v in d["tracking_timestamp_us"]], dtype=np.int64)
    p_dev = np.stack([d[f"t{a}_odometry_device"] for a in "xyz"], -1)
    R_w_dev = np.stack([vibaio.quat_to_rot([d["qw_odometry_device"][k], d["qx_odometry_device"][k],
                                            d["qy_odometry_device"][k], d["qz_odometry_device"][k]])
                        for k in range(len(kf_ns))])
    v_dev = np.stack([d[f"device_linear_velocity_{a}_odometry"] for a in "xyz"], -1)
    w_dev = np.stack([d[f"angular_velocity_{a}_device"] for a in "xyz"], -1)
    R_w_imu = R_w_dev @ T_dev_imu[:3, :3]
    p_imu = p_dev + R_w_dev @ T_dev_imu[:3, 3]
    v_imu = v_dev + np.einsum("nij,nj->ni", R_w_dev, np.cross(w_dev, T_dev_imu[:3, 3]))
    N = len(kf_ns)
    write_traj(dest / "traj_kf.txt", kf_ns, p_imu, R_w_imu)

    # ---- calibration per keyframe
    calibs = [json.loads(ln) for ln in (outd / "online_calibration.jsonl").read_text().splitlines() if ln.strip()]
    assert len(calibs) == N and all(int(c["tracking_timestamp_us"]) == kf_us[i] for i, c in enumerate(calibs))
    imus = [imu_params(next(x for x in c["ImuCalibrations"] if x["Label"] == IMU_LABEL)) for c in calibs]

    # ---- raw IMU
    imu_file = inp / f"imu_samples_{IMU_LABEL}.csv"
    t_imu = np.loadtxt(imu_file, delimiter=",", skiprows=1, usecols=0, dtype=np.int64)
    raw = np.loadtxt(imu_file, delimiter=",", skiprows=1, usecols=(2, 3, 4, 5, 6, 7))
    gyro_raw, acc_raw = raw[:, :3], raw[:, 3:]

    # ---- dense trajectory at the camera frames
    frames = np.loadtxt(frames_csv, delimiter=",", usecols=0, dtype=np.int64, comments="#")
    frames = frames[(frames >= kf_ns[0]) & (frames <= kf_ns[-1])]
    dense_t, dense_p, dense_R = [], [], []
    end_dp, end_dr, end_dv = [], [], []
    for i in range(N - 1):
        c = imus[i]  # VI-BA preintegrates the interval with the calibration of its first keyframe
        T0, T1 = int(kf_ns[i]), int(kf_ns[i + 1])
        tg = t_imu - int(round(c["dtg"] * 1e9))  # reference-clock end of each sample's hold interval
        ta = t_imu - int(round(c["dta"] * 1e9))
        fr = frames[(frames > T0) & (frames < T1)]
        cuts = np.unique(np.concatenate([[T0, T1], fr, tg[(tg > T0) & (tg < T1)], ta[(ta > T0) & (ta < T1)]]))
        mid = 0.5 * (cuts[:-1] + cuts[1:])
        gi = np.searchsorted(tg, mid, side="left")  # sample k covers (t[k-1], t[k]]
        ai = np.searchsorted(ta, mid, side="left")
        w = gyro_raw[gi] @ np.linalg.inv(c["Sg"]).T - c["bg"]
        a = acc_raw[ai] @ np.linalg.inv(c["Sa"]).T - c["ba"]
        dts = np.diff(cuts) * 1e-9
        R, p, v = R_w_imu[i].copy(), p_imu[i].copy(), v_imu[i].copy()
        seg_t, seg_p, seg_R = [], [], []
        frset = set(fr.tolist())
        for k in range(len(dts)):
            dt = dts[k]
            Rm = R @ vibaio.so3_exp(w[k] * (0.5 * dt))
            aw = Rm @ a[k] + GRAVITY
            p = p + v * dt + 0.5 * aw * dt * dt
            v = v + aw * dt
            R = R @ vibaio.so3_exp(w[k] * dt)
            if int(cuts[k + 1]) in frset:
                seg_t.append(int(cuts[k + 1])); seg_p.append(p.copy()); seg_R.append(R.copy())
        ep = p_imu[i + 1] - p
        er = vibaio.so3_log(R.T @ R_w_imu[i + 1])
        end_dp.append(np.linalg.norm(ep)); end_dr.append(np.linalg.norm(er)); end_dv.append(np.linalg.norm(v_imu[i + 1] - v))
        dense_t.append(T0); dense_p.append(p_imu[i]); dense_R.append(R_w_imu[i])
        for t, pp, RR in zip(seg_t, seg_p, seg_R):
            s = (t - T0) / (T1 - T0)
            dense_t.append(t); dense_p.append(pp + s * ep); dense_R.append(RR @ vibaio.so3_exp(s * er))
    dense_t.append(int(kf_ns[-1])); dense_p.append(p_imu[-1]); dense_R.append(R_w_imu[-1])
    write_traj(dest / "traj.txt", dense_t, dense_p, dense_R)
    end_dp, end_dr, end_dv = np.array(end_dp), np.array(end_dr), np.array(end_dv)
    chk = (f"IMU integration keyframe to keyframe ({N - 1} intervals, median {np.median(np.diff(kf_ns)) * 1e-9:.2f} s) vs "
           f"optimized next keyframe:\n"
           f"  position mismatch median {1e3 * np.median(end_dp):.2f} mm, p90 {1e3 * np.percentile(end_dp, 90):.2f} mm, max {1e3 * end_dp.max():.2f} mm\n"
           f"  rotation mismatch median {np.degrees(np.median(end_dr)):.4f} deg, p90 {np.degrees(np.percentile(end_dr, 90)):.4f} deg, max {np.degrees(end_dr.max()):.4f} deg\n"
           f"  velocity mismatch median {1e3 * np.median(end_dv):.2f} mm/s, p90 {1e3 * np.percentile(end_dv, 90):.2f} mm/s, max {1e3 * end_dv.max():.2f} mm/s\n"
           f"  dense trajectory: {len(dense_t)} poses\n")
    (dest / "densify_check.txt").write_text(chk)
    print(chk, end="")

    # ---- calibration windows: runs of keyframes sharing one calibration variable
    def key(c):
        return json.dumps([c["ImuCalibrations"], [(x["Projection"]["Params"], x["T_Device_Camera"]) for x in c["CameraCalibrations"]]],
                          sort_keys=True)

    keys = [key(c) for c in calibs]
    starts = [0] + [i for i in range(1, N) if keys[i] != keys[i - 1]]
    ends = starts[1:] + [N]
    t0_ns = int(kf_ns[0])
    # sensor temperature of camera-slam-left from the VRS image metadata (SLAM Dunk's extraction: cam0/meta.csv), if there
    meta = None
    for cand in (frames_csv.parent / "meta.csv", inp / "cam0_meta.csv"):
        if cand.exists():
            meta = np.loadtxt(cand, delimiter=",", comments="#")
            break
    rows = []
    prev = None
    for s, e in zip(starts, ends):
        c = calibs[s]
        m = imus[s]
        t_first = (kf_ns[s] - t0_ns) * 1e-9
        t_last = (kf_ns[e - 1] - t0_ns) * 1e-9
        t_mean_us = kf_us[s:e].mean()  # VI-BA's window time (average keyframe timestamp)
        t_mean = (t_mean_us * 1000 - t0_ns) * 1e-9
        # mean factory-rectified specific force / rate over the window [first keyframe, first keyframe of next window)
        hi_ns = kf_ns[e] if e < N else kf_ns[e - 1]
        sel = slice(np.searchsorted(t_imu, kf_ns[s]), max(np.searchsorted(t_imu, hi_ns), np.searchsorted(t_imu, kf_ns[s]) + 1))
        af_mean = ((acc_raw[sel] - fp["ba"]) @ Ma_inv.T).mean(0)
        wf_mean = ((gyro_raw[sel] - fp["bg"]) @ Mg_inv.T).mean(0)
        Sa_inv, Sg_inv = np.linalg.inv(m["Sa"]), np.linalg.inv(m["Sg"])
        # bias a bias-only model on factory-rectified samples would need: b' = a_f - a_true, a_true = S^-1 raw - b
        ba_fr = m["ba"] - Sa_inv @ fp["ba"]
        bg_fr = m["bg"] - Sg_inv @ fp["bg"]
        ba_eff = (np.eye(3) - Sa_inv @ fp["Sa"]) @ af_mean + ba_fr
        bg_eff = (np.eye(3) - Sg_inv @ fp["Sg"]) @ wf_mean + bg_fr
        a_scale = np.linalg.norm(m["Sa"], axis=1)
        g_scale = np.linalg.norm(m["Sg"], axis=1)
        a_no = m["Sa"] / a_scale[:, None]
        g_no = m["Sg"] / g_scale[:, None]
        row = dict(window=len(rows), n_kf=e - s, t_first_s=t_first, t_last_s=t_last, t_mean_s=t_mean,
                   t_mean_device_s=t_mean_us * 1e-6)
        if meta is not None:
            mm = (meta[:, 0] >= kf_ns[s]) & (meta[:, 0] <= kf_ns[e - 1])
            row["temp_cam_left_C"] = meta[mm, 3].mean() if mm.any() else float("nan")
        else:
            row["temp_cam_left_C"] = float("nan")
        for k, ax in enumerate("xyz"):
            row[f"ba_{ax}"] = m["ba"][k]
        for k, ax in enumerate("xyz"):
            row[f"bg_{ax}"] = m["bg"][k]
        for k, ax in enumerate("xyz"):
            row[f"ba_fr_{ax}"] = ba_fr[k]
        for k, ax in enumerate("xyz"):
            row[f"bg_fr_{ax}"] = bg_fr[k]
        for k, ax in enumerate("xyz"):
            row[f"ba_eff_{ax}"] = ba_eff[k]
        for k, ax in enumerate("xyz"):
            row[f"bg_eff_{ax}"] = bg_eff[k]
        for k, ax in enumerate("xyz"):
            row[f"acc_scale_{ax}"] = a_scale[k]
        row.update(acc_nonorth_xy=a_no[0, 1], acc_nonorth_xz=a_no[0, 2], acc_nonorth_yz=a_no[1, 2])
        for k, ax in enumerate("xyz"):
            row[f"gyr_scale_{ax}"] = g_scale[k]
        for (r_, c_) in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]:
            row[f"gyr_nonorth_{'xyz'[r_]}{'xyz'[c_]}"] = g_no[r_, c_]
        row["dt_gyro_ms"] = 1e3 * m["dtg"]
        row["dt_accel_ms"] = 1e3 * m["dta"]
        # normalized random-walk residuals wrt the previous window, for VI-BA's default (uninflated) bias walks
        if prev is None:
            z_a = z_g = np.full(3, np.nan)
        else:
            dt = (t_mean_us - prev["t_us"]) * 1e-6
            z_a = (m["ba"] - prev["ba"]) / np.sqrt(ACC_BIAS_RW_VAR * dt)
            z_g = (m["bg"] - prev["bg"]) / np.sqrt(GYR_BIAS_RW_VAR * dt)
        for k, ax in enumerate("xyz"):
            row[f"rwz_ba_{ax}"] = z_a[k]
        for k, ax in enumerate("xyz"):
            row[f"rwz_bg_{ax}"] = z_g[k]
        prev = dict(t_us=t_mean_us, ba=m["ba"], bg=m["bg"])
        for ci, lab in enumerate(CAM_LABELS):
            cj = next(x for x in c["CameraCalibrations"] if x["Label"] == lab)
            prm = np.array(cj["Projection"]["Params"])
            fprm = np.array(fac_cams[ci]["Projection"]["Params"])
            for k, nm in enumerate(INTR_NAMES):
                row[f"{CAM_SHORT[ci]}_{nm}"] = prm[k]
            for k, nm in enumerate(INTR_NAMES[:3]):
                row[f"{CAM_SHORT[ci]}_d{nm}"] = prm[k] - fprm[k]
            T_imu_cam = T_imu_dev @ T_json(cj["T_Device_Camera"])
            rv = vibaio.so3_log(T_imu_cam_fac[ci][:3, :3].T @ T_imu_cam[:3, :3])
            dt_ = T_imu_cam[:3, 3] - T_imu_cam_fac[ci][:3, 3]
            for k, ax in enumerate("xyz"):
                row[f"{CAM_SHORT[ci]}_drot_{ax}_mrad"] = 1e3 * rv[k]
            for k, ax in enumerate("xyz"):
                row[f"{CAM_SHORT[ci]}_dpos_{ax}_mm"] = 1e3 * dt_[k]
        rows.append(row)

    fa = Ma_inv @ fp["ba"]
    header = f"""# VI-BA calibration estimates per calibration window (windows of ~5 s; one optimization variable per window and sensor).
# source: {outd}
# time: t_first_s/t_last_s = first/last keyframe of the window, t_mean_s = mean keyframe time (the time VI-BA assigns to the
#   window in its random walks), all relative to the first keyframe (device clock {stamp(t0_ns)} s); t_mean_device_s = device clock.
# temp_cam_left_C: mean sensor temperature of camera-slam-left over the window (VRS image metadata, 1 degC resolution; the IMU
#   samples of these recordings carry no temperature); nan if not available.
# IMU = imu-right (the body frame), all vectors in the imu-right sensor frame. VI-BA's model:
#   true = S^-1 * raw - b   (raw = unrectified VRS sample; S = diag(scale) * nonorth, rows of nonorth have unit norm,
#   the accelerometer's is upper triangular), i.e. b = (compensated measurement) - (true value).
#   ba_* [m/s^2], bg_* [rad/s]: VI-BA's bias states b.
#   ba_fr_*, bg_fr_*: b - S^-1 * b_factory: the bias left on samples from which the factory bias was already removed
#     (factory, Aria convention: raw = M * rectified + b_factory), keeping VI-BA's S for the scale part.
#   ba_eff_*, bg_eff_*: bias that a bias-only model applied to factory-rectified samples would need in this window,
#     = (factory-rectified mean signal of the window) - (VI-BA-compensated mean signal): it adds the effect of the scale /
#     non-orthogonality difference to the factory matrix at the window's mean specific force / rate. Same sign convention.
#   acc_scale_*, gyr_scale_*: diagonal of S (row norms); acc_nonorth_ij, gyr_nonorth_ij: off-diagonal elements (row i, column j)
#     of the unit-row matrix nonorth (~ sine of the misalignment angles).
#   dt_gyro_ms, dt_accel_ms: TimeOffsetSec_Device_Gyro / _Accel: a sample stamped t refers to t - dt on the camera clock.
#   rwz_ba_*, rwz_bg_*: change of the bias from the previous window divided by the standard deviation VI-BA's default random
#     walk allows over that time (sqrt(1e-8 m^2/s^4 /s * dt), sqrt(1e-10 rad^2/s^2 /s * dt)); nan in the first window.
# cameras: cl = camera-slam-left, cr = camera-slam-right, FisheyeRadTanThinPrism (Fisheye624) on the 640x480 image:
#   f, cu, cv [px] (pixel (0,0) = centre of the top-left pixel), k0..k5, p0, p1, s0..s3; df, dcu, dcv = estimate - factory [px].
#   drot_*_mrad: rotation vector of R_factory^T * R_estimate with R = R_imu_camera (camera axes in the imu-right frame),
#     components in the factory camera frame; dpos_*_mm: (camera centre in the imu-right frame) estimate - factory.
# factory values: accel bias (Aria convention) {fp['ba'].tolist()}, as VI-BA-convention bias M^-1 b: {fa.tolist()}
#   gyro bias {fp['bg'].tolist()}; accel M diag {np.diag(fp['Sa']).tolist()}, gyro M diag {np.diag(fp['Sg']).tolist()}
#   dt_gyro {1e3 * fp['dtg']:.6f} ms, dt_accel {1e3 * fp['dta']:.6f} ms
#   cl: {fac_cams[0]['Projection']['Params']}
#   cr: {fac_cams[1]['Projection']['Params']}
"""
    cols = list(rows[0].keys())
    # the factory calibration in the same columns (as a comment line, so that plain csv readers skip it)
    fr = {c: float("nan") for c in cols}
    fs_a, fs_g = np.linalg.norm(fp["Sa"], axis=1), np.linalg.norm(fp["Sg"], axis=1)
    fn_a, fn_g = fp["Sa"] / fs_a[:, None], fp["Sg"] / fs_g[:, None]
    for k, ax in enumerate("xyz"):
        fr[f"ba_{ax}"], fr[f"bg_{ax}"] = fa[k], (Mg_inv @ fp["bg"])[k]
        fr[f"ba_fr_{ax}"] = fr[f"bg_fr_{ax}"] = fr[f"ba_eff_{ax}"] = fr[f"bg_eff_{ax}"] = 0.0
        fr[f"acc_scale_{ax}"], fr[f"gyr_scale_{ax}"] = fs_a[k], fs_g[k]
    fr.update(acc_nonorth_xy=fn_a[0, 1], acc_nonorth_xz=fn_a[0, 2], acc_nonorth_yz=fn_a[1, 2],
              dt_gyro_ms=1e3 * fp["dtg"], dt_accel_ms=1e3 * fp["dta"])
    for (r_, c_) in [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]:
        fr[f"gyr_nonorth_{'xyz'[r_]}{'xyz'[c_]}"] = fn_g[r_, c_]
    for ci in range(2):
        for k, nm in enumerate(INTR_NAMES):
            fr[f"{CAM_SHORT[ci]}_{nm}"] = fac_cams[ci]["Projection"]["Params"][k]
        for nm in INTR_NAMES[:3]:
            fr[f"{CAM_SHORT[ci]}_d{nm}"] = 0.0
        for ax in "xyz":
            fr[f"{CAM_SHORT[ci]}_drot_{ax}_mrad"] = fr[f"{CAM_SHORT[ci]}_dpos_{ax}_mm"] = 0.0
    header += "# the next line is the factory calibration in the columns of the table (nan where not applicable):\n"
    header += "#factory_row," + ",".join(f"{fr[c]:.10g}" for c in cols[1:]) + "\n"
    with open(dest / "calib_vs_time.csv", "w") as f:
        f.write(header)
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[c]) if isinstance(r[c], (int, np.integer)) else f"{r[c]:.10g}" for c in cols) + "\n")
    print(f"{len(rows)} calibration windows -> {dest / 'calib_vs_time.csv'}")


if __name__ == "__main__":
    main()
