# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers of the converters that write VIBA input folders (see README.md).

A VIBA input folder (read by interfaces/ark/session_data/SessionData.cpp) contains:

  factory_calibration.json   cameras (model, intrinsics, T_Device_Camera) and IMUs
  online_calibration.jsonl   one calibration per keyframe (here: the factory one, repeated)
  open_loop_trajectory.csv   keyframe states T_world_device, velocity, angular velocity (initial guess)
  session_observations.csv   2d observations: point id, keyframe timestamp [us], camera, pixel, sqrt info
  imu_samples_<label>.csv    IMU samples
  vrs_source_info.json       camera / IMU labels
  imu_noise.json             IMU noise model (optional)

Conventions: device frame = IMU frame; pixel (0, 0) is the center of the top-left pixel (Kalibr,
OpenCV, Aria; COLMAP uses (0.5, 0.5), converters must subtract 0.5); quaternions are (w, x, y, z)
unless named otherwise.
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError:  # only needed for Kalibr / ASL yaml files
    yaml = None

# ------------------------------------------------------------------------------------------------
# rotations (quaternions as w, x, y, z)


def quat_to_rot(q):
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def rot_to_quat(R):
    """Shepperd's method, returns w, x, y, z with w >= 0."""
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * np.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.array(q)
    return q if q[0] >= 0 else -q


def se3_json(R, t):
    q = rot_to_quat(R)
    return {"Translation": [float(v) for v in t], "UnitQuaternion": [float(q[0]), [float(v) for v in q[1:]]]}


def invert(T):
    Ti = np.eye(4)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def slerp(q0, q1, a):
    """Quaternion (w, x, y, z) interpolation, a in [0, 1]."""
    q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
    d = float(q0 @ q1)
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + a * (q1 - q0)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - a) * th) * q0 + np.sin(a * th) * q1) / np.sin(th)
    return q / np.linalg.norm(q)


def so3_log(R):
    c = np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)
    th = np.arccos(c)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    if th < 1e-8:
        return 0.5 * w
    return th / (2 * np.sin(th)) * w


def so3_exp(w):
    th = np.linalg.norm(w)
    K = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
    if th < 1e-8:
        return np.eye(3) + K
    return np.eye(3) + np.sin(th) / th * K + (1 - np.cos(th)) / th**2 * K @ K


# ------------------------------------------------------------------------------------------------
# camera models: names of lib/calibration (CameraProjection.cpp) and numpy projections


MODEL_DESCRIPTIONS = {
    "Linear": "fx, fy, cx, cy",
    "KannalaBrandtK3": "fx, fy, cx, cy, kb0, kb1, kb2, kb3",
    "FisheyeRadTanThinPrism": "fx, cx, cy, k_0, k_1, k_2, k_3, k_4, k_5, p_0, p_1, s_0, s_1, s_2, s_3",
    "PinholeRadTan": "fx, fy, cx, cy, k1, k2, p1, p2",
    "PinholeRadTanK3": "fx, fy, cx, cy, k1, k2, p1, p2, k3",
    "DoubleSphere": "fx, fy, cx, cy, xi, alpha",
    "ExtendedUnified": "fx, fy, cx, cy, alpha, beta",
    "Unified": "fx, fy, cx, cy, alpha",
}
FISHEYE_MODELS = {"KannalaBrandtK3", "FisheyeRadTanThinPrism", "DoubleSphere", "ExtendedUnified", "Unified"}


def project(model, params, P):
    """Projects points P (N x 3, camera frame) to pixels (N x 2)."""
    P = np.atleast_2d(np.asarray(P, float))
    x, y, z = P[:, 0], P[:, 1], P[:, 2]
    p = np.asarray(params, float)
    if model == "FisheyeRadTanThinPrism":
        f, cx, cy, k, tp, s = p[0], p[1], p[2], p[3:9], p[9:11], p[11:15]
        r = np.hypot(x, y)
        th = np.arctan2(r, z)
        th2 = th * th
        rd = th * (1 + sum(k[i] * th2 ** (i + 1) for i in range(6)))
        sc = np.where(r > 1e-12, rd / np.maximum(r, 1e-300), 1 / z)
        a, b = x * sc, y * sc
        r2 = a * a + b * b
        # Aria's tangential convention: (2 (a p0 + b p1)) (a, b) + r2 (p0, p1)
        u = a + tp[0] * (r2 + 2 * a * a) + 2 * tp[1] * a * b + s[0] * r2 + s[1] * r2 * r2
        v = b + 2 * tp[0] * a * b + tp[1] * (r2 + 2 * b * b) + s[2] * r2 + s[3] * r2 * r2
        return np.stack([f * u + cx, f * v + cy], -1)
    fx, fy, cx, cy = p[:4]
    if model == "Linear":
        mx, my = x / z, y / z
    elif model == "KannalaBrandtK3":
        r = np.hypot(x, y)
        th = np.arctan2(r, z)
        th2 = th * th
        rd = th * (1 + th2 * (p[4] + th2 * (p[5] + th2 * (p[6] + th2 * p[7]))))
        sc = np.where(r > 1e-12, rd / np.maximum(r, 1e-300), 1 / z)
        mx, my = x * sc, y * sc
    elif model in ("PinholeRadTan", "PinholeRadTanK3"):
        k1, k2, p1, p2 = p[4:8]
        k3 = p[8] if model == "PinholeRadTanK3" else 0.0
        a, b = x / z, y / z
        r2 = a * a + b * b
        rad = 1 + r2 * (k1 + r2 * (k2 + r2 * k3))
        mx = a * rad + 2 * p1 * a * b + p2 * (r2 + 2 * a * a)
        my = b * rad + p1 * (r2 + 2 * b * b) + 2 * p2 * a * b
    else:
        if model == "DoubleSphere":
            xi, al = p[4], p[5]
            d1 = np.sqrt(x * x + y * y + z * z)
            zeta = xi * d1 + z
            den = al * np.sqrt(x * x + y * y + zeta * zeta) + (1 - al) * zeta
        elif model == "ExtendedUnified":
            al, be = p[4], p[5]
            den = al * np.sqrt(be * (x * x + y * y) + z * z) + (1 - al) * z
        elif model == "Unified":
            al = p[4]
            den = al * np.sqrt(x * x + y * y + z * z) + (1 - al) * z
        else:
            raise ValueError(f"unknown camera model {model}")
        mx, my = x / den, y / den
    return np.stack([fx * mx + cx, fy * my + cy], -1)


def opencv_undistort_points(params, uv, iters=5):
    """Emulates cv2.undistortPoints for the radtan models (fixed-point iteration, OpenCV's default
    is 5 iterations, which is not converged near the image border on EuRoC: up to 0.3 px)."""
    fx, fy, cx, cy, k1, k2, p1, p2 = params[:8]
    k3 = params[8] if len(params) > 8 else 0.0
    x0 = (uv[:, 0] - cx) / fx
    y0 = (uv[:, 1] - cy) / fy
    x, y = x0.copy(), y0.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        icdist = 1 / (1 + ((k3 * r2 + k2) * r2 + k1) * r2)
        dx = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        dy = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
        x = (x0 - dx) * icdist
        y = (y0 - dy) * icdist
    return np.stack([x, y], -1)


def pixels_from_normalized(cam, xy, opencv_iters=5):
    """Pixels whose undistortion gave the normalized coordinates `xy` (N x 2).

    For radtan models with opencv_iters > 0 the (unconverged) cv2.undistortPoints map is inverted
    with Newton's method, which recovers the measured pixels exactly; otherwise the exact
    projection is used (also right for cv2.fisheye.undistortPoints, which converges)."""
    xy = np.asarray(xy, float)
    uv = cam.project(np.c_[xy, np.ones(len(xy))])
    if cam.model not in ("PinholeRadTan", "PinholeRadTanK3") or opencv_iters <= 0:
        return uv
    h = 1e-3
    for _ in range(20):
        r = opencv_undistort_points(cam.params, uv, opencv_iters) - xy
        if np.abs(r).max() < 1e-13:
            break
        ju = (opencv_undistort_points(cam.params, uv + [h, 0], opencv_iters) - xy - r) / h
        jv = (opencv_undistort_points(cam.params, uv + [0, h], opencv_iters) - xy - r) / h
        det = ju[:, 0] * jv[:, 1] - jv[:, 0] * ju[:, 1]
        du = (jv[:, 1] * r[:, 0] - jv[:, 0] * r[:, 1]) / det
        dv = (-ju[:, 1] * r[:, 0] + ju[:, 0] * r[:, 1]) / det
        uv = uv - np.stack([du, dv], -1)
    return uv


@dataclass
class Camera:
    label: str
    model: str  # lib/calibration name, see MODEL_DESCRIPTIONS
    params: list
    T_imu_cam: np.ndarray  # 4x4, device (= IMU) from camera
    width: int
    height: int
    valid_radius: float = None
    max_solid_angle: float = None
    time_offset_sec: float = 0.0  # TimeOffsetSec_Device_Camera

    def to_json(self):
        if self.model not in MODEL_DESCRIPTIONS:
            raise ValueError(f"camera {self.label}: unknown model {self.model}")
        max_angle = self.max_solid_angle
        if max_angle is None:  # only used by checked projection
            if self.model in FISHEYE_MODELS:
                max_angle = 1.7
            else:
                max_angle = min(1.5, 1.2 * np.arctan(0.5 * np.hypot(self.width, self.height) / self.params[0]))
        config = {"ImageWidth": int(self.width), "ImageHeight": int(self.height), "MaxSolidAngle": float(max_angle)}
        if self.valid_radius is not None:
            config["ValidRadius"] = float(self.valid_radius)
        j = {
            "Calibrated": True,
            "Label": self.label,
            "SerialNumber": self.label,
            "T_Device_Camera": se3_json(self.T_imu_cam[:3, :3], self.T_imu_cam[:3, 3]),
            "Projection": {
                "Name": self.model,
                "Description": MODEL_DESCRIPTIONS[self.model],
                "Params": [float(v) for v in self.params],
            },
            "ConfigData": config,
        }
        if self.time_offset_sec:
            j["TimeOffsetSec_Device_Camera"] = float(self.time_offset_sec)
        return j

    def project(self, P):
        return project(self.model, self.params, P)


@dataclass
class ImuNoise:
    """Continuous-time noise densities (Kalibr convention)."""

    rate_hz: float
    acc_noise_density: float  # m/s^1.5
    acc_random_walk: float  # m/s^2.5
    gyr_noise_density: float  # rad/s^0.5
    gyr_random_walk: float  # rad/s^1.5

    def to_json(self, label):
        return {
            label: {
                "accel_sample_variance": self.acc_noise_density**2 * self.rate_hz,
                "gyro_sample_variance": self.gyr_noise_density**2 * self.rate_hz,
                "accel_bias_random_walk_variance_per_sec": self.acc_random_walk**2,
                "gyro_bias_random_walk_variance_per_sec": self.gyr_random_walk**2,
            }
        }


@dataclass
class Imu:
    label: str = "imu0"
    accel_bias: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    gyro_bias: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def to_json(self):
        return {
            "Label": self.label,
            "SerialNumber": self.label,
            "Calibrated": True,
            "T_Device_Imu": se3_json(np.eye(3), np.zeros(3)),
            "Accelerometer": {
                "Model": {"Name": "UpperTriagonalLinear", "RectificationMatrix": np.eye(3).tolist()},
                "Bias": {"Name": "Constant", "Offset": [float(v) for v in self.accel_bias]},
                "TimeOffsetSec_Device_Accel": 0.0,
            },
            "Gyroscope": {
                "Model": {"Name": "Linear", "RectificationMatrix": np.eye(3).tolist()},
                "Bias": {"Name": "Constant", "Offset": [float(v) for v in self.gyro_bias]},
                "TimeOffsetSec_Device_Gyro": 0.0,
            },
        }


# ------------------------------------------------------------------------------------------------
# calibration readers


def _need_yaml():
    if yaml is None:
        raise SystemExit("PyYAML is required to read Kalibr / ASL yaml files (pip install pyyaml)")


def camera_from_kalibr(label, c):
    """One camera of a Kalibr camchain (camera_model / distortion_model / intrinsics / ...)."""
    cm, dm = c["camera_model"], c.get("distortion_model", "none")
    intr, dist = list(c["intrinsics"]), list(c.get("distortion_coeffs", []))
    if cm == "pinhole" and dm == "equidistant":
        model, params = "KannalaBrandtK3", intr + dist
    elif cm == "pinhole" and dm in ("radtan", "radial-tangential"):
        model, params = "PinholeRadTan", intr + dist
    elif cm == "pinhole" and dm == "none":
        model, params = "Linear", intr
    elif cm == "ds" and dm == "none":  # intrinsics: xi, alpha, fu, fv, pu, pv
        model, params = "DoubleSphere", intr[2:6] + intr[0:2]
    elif cm == "eucm" and dm == "none":  # intrinsics: alpha, beta, fu, fv, pu, pv
        model, params = "ExtendedUnified", intr[2:6] + intr[0:2]
    elif cm == "omni" and dm == "none":  # Mei: xi, gamma1, gamma2, pu, pv
        xi = intr[0]
        model, params = "Unified", [intr[1] / (1 + xi), intr[2] / (1 + xi), intr[3], intr[4], xi / (1 + xi)]
    else:
        raise ValueError(f"{label}: unsupported Kalibr camera {cm}-{dm}")
    T_imu_cam = invert(np.array(c["T_cam_imu"], float))
    w, h = c["resolution"]
    # timeshift_cam_imu is not applied: datasets (EuRoC, TUM-VI) ship time-corrected stamps
    return Camera(label, model, params, T_imu_cam, w, h)


def read_kalibr_camchain(path):
    _need_yaml()
    chain = yaml.safe_load(Path(path).read_text())
    return [camera_from_kalibr(name, chain[name]) for name in sorted(chain) if name.startswith("cam")]


def read_asl_cameras(mav0):
    """EuRoC-style mav0/camX/sensor.yaml (T_BS = body from sensor, body = IMU for EuRoC)."""
    _need_yaml()
    cams = []
    for d in sorted(Path(mav0).glob("cam*")):
        f = d / "sensor.yaml"
        if not f.exists():
            continue
        s = yaml.safe_load(f.read_text())
        T_BS = np.array(s["T_BS"]["data"], float).reshape(4, 4)
        dm = {"radial-tangential": "radtan", "equidistant": "equidistant"}.get(s["distortion_model"], s["distortion_model"])
        c = {
            "camera_model": s["camera_model"],
            "distortion_model": dm,
            "intrinsics": s["intrinsics"],
            "distortion_coeffs": s["distortion_coefficients"],
            "T_cam_imu": invert(T_BS).tolist(),
            "resolution": s["resolution"],
        }
        cams.append(camera_from_kalibr(d.name, c))
    imu_yaml = Path(mav0) / "imu0" / "sensor.yaml"
    if imu_yaml.exists():
        T_BI = np.array(yaml.safe_load(imu_yaml.read_text())["T_BS"]["data"], float).reshape(4, 4)
        if not np.allclose(T_BI, np.eye(4)):  # express cameras in the IMU frame
            for c in cams:
                c.T_imu_cam = invert(T_BI) @ c.T_imu_cam
    return cams


BASALT_MODELS = {
    # basalt camera_type: (lib/calibration name, intrinsics keys)
    "pinhole": ("Linear", ["fx", "fy", "cx", "cy"]),
    "kb4": ("KannalaBrandtK3", ["fx", "fy", "cx", "cy", "k1", "k2", "k3", "k4"]),
    "ds": ("DoubleSphere", ["fx", "fy", "cx", "cy", "xi", "alpha"]),
    "eucm": ("ExtendedUnified", ["fx", "fy", "cx", "cy", "alpha", "beta"]),
    "ucm": ("Unified", ["fx", "fy", "cx", "cy", "alpha"]),
}


def read_basalt_calib(path):
    """Basalt calibration json; returns (cameras, Imu with Basalt's static biases, ImuNoise or None)."""
    v = json.loads(Path(path).read_text())["value0"]
    cams = []
    for i, (intr, T, res) in enumerate(zip(v["intrinsics"], v["T_imu_cam"], v["resolution"])):
        t = intr["camera_type"]
        if t not in BASALT_MODELS:
            raise ValueError(f"cam{i}: unsupported Basalt camera type {t}")
        model, keys = BASALT_MODELS[t]
        params = [intr["intrinsics"][k] for k in keys]
        R = quat_to_rot([T["qw"], T["qx"], T["qy"], T["qz"]])
        cams.append(Camera(f"cam{i}", model, params, make_T(R, [T["px"], T["py"], T["pz"]]), res[0], res[1]))
    # basalt's calib_*_bias also hold scale / misalignment terms; only the biases are used here
    imu = Imu(accel_bias=list(v["calib_accel_bias"][:3]), gyro_bias=list(v["calib_gyro_bias"][:3]))
    noise = None
    if "accel_noise_std" in v:
        noise = ImuNoise(
            rate_hz=v["imu_update_rate"],
            acc_noise_density=v["accel_noise_std"][0],
            acc_random_walk=v["accel_bias_std"][0],
            gyr_noise_density=v["gyro_noise_std"][0],
            gyr_random_walk=v["gyro_bias_std"][0],
        )
    return cams, imu, noise


def read_kalibr_imu_noise(path, rate_hz=None):
    """Kalibr imu yaml (also EuRoC's mav0/imu0/sensor.yaml)."""
    _need_yaml()
    s = yaml.safe_load(Path(path).read_text())
    return ImuNoise(
        rate_hz=rate_hz or float(s.get("update_rate", s.get("rate_hz"))),
        acc_noise_density=float(s["accelerometer_noise_density"]),
        acc_random_walk=float(s["accelerometer_random_walk"]),
        gyr_noise_density=float(s["gyroscope_noise_density"]),
        gyr_random_walk=float(s["gyroscope_random_walk"]),
    )


def colmap_camera(label, model, params, width, height):
    """COLMAP camera -> Camera with identity extrinsics. COLMAP's pixel (0.5, 0.5) is the center of
    the top-left pixel, the principal point is shifted by -0.5 here (observations must be too)."""
    p = list(map(float, params))

    def zeros(v):
        return all(abs(x) < 1e-15 for x in v)

    if model == "SIMPLE_PINHOLE":
        m, q = "Linear", [p[0], p[0], p[1], p[2]]
    elif model == "PINHOLE":
        m, q = "Linear", p[:4]
    elif model == "SIMPLE_RADIAL":
        m, q = "PinholeRadTan", [p[0], p[0], p[1], p[2], p[3], 0.0, 0.0, 0.0]
    elif model == "RADIAL":
        m, q = "PinholeRadTan", [p[0], p[0], p[1], p[2], p[3], p[4], 0.0, 0.0]
    elif model == "OPENCV":
        m, q = "PinholeRadTan", p[:8]
    elif model == "FULL_OPENCV":  # k1 k2 p1 p2 k3 k4 k5 k6 (rational model)
        if not zeros(p[9:12]):
            raise ValueError("FULL_OPENCV with k4..k6 != 0 (rational model) is not supported")
        m, q = "PinholeRadTanK3", p[:9]
    elif model == "OPENCV_FISHEYE":
        m, q = "KannalaBrandtK3", p[:8]
    elif model == "SIMPLE_RADIAL_FISHEYE":
        m, q = "KannalaBrandtK3", [p[0], p[0], p[1], p[2], p[3], 0.0, 0.0, 0.0]
    elif model == "RADIAL_FISHEYE":
        m, q = "KannalaBrandtK3", [p[0], p[0], p[1], p[2], p[3], p[4], 0.0, 0.0]
    elif model == "RAD_TAN_THIN_PRISM_FISHEYE":  # fx fy cx cy k0..k5 p0 p1 s0..s3
        if zeros(p[8:16]):
            m, q = "KannalaBrandtK3", p[:8]
        elif abs(p[0] - p[1]) < 1e-6 * p[0]:
            m, q = "FisheyeRadTanThinPrism", [p[0]] + p[2:16]
        else:
            raise ValueError("RAD_TAN_THIN_PRISM_FISHEYE with fx != fy is not supported")
    else:
        raise ValueError(f"unsupported COLMAP camera model {model}")
    pp = (1, 2) if m == "FisheyeRadTanThinPrism" else (2, 3)
    q[pp[0]] -= 0.5
    q[pp[1]] -= 0.5
    return Camera(label, m, q, np.eye(4), width, height)


def read_calibration(path):
    """Kalibr camchain (.yaml), Basalt calibration (.json), or an ASL mav0 folder.
    Returns (cameras, Imu or None, ImuNoise or None)."""
    path = Path(path)
    if path.is_dir():
        return read_asl_cameras(path), None, None
    if path.suffix == ".json":
        return read_basalt_calib(path)
    return read_kalibr_camchain(path), None, None


def dataset_calibration(dataset):
    """Calibration of an ASL dataset folder (TUM-VI: dso/camchain.yaml, EuRoC: mav0/cam*/sensor.yaml)
    and its IMU noise (TUM-VI: dso/imu_config.yaml, EuRoC: mav0/imu0/sensor.yaml)."""
    dataset = Path(dataset)
    if (dataset / "dso" / "camchain.yaml").exists():
        cams = read_kalibr_camchain(dataset / "dso" / "camchain.yaml")
    else:
        cams = read_asl_cameras(dataset / "mav0")
    noise = None
    for f in [dataset / "dso" / "imu_config.yaml", dataset / "mav0" / "imu0" / "sensor.yaml"]:
        if f.exists():
            noise = read_kalibr_imu_noise(f)
            break
    return cams, noise


def is_tumvi(dataset):
    return (Path(dataset) / "dso" / "camchain.yaml").exists()


def default_valid_radius(dataset, cams):
    """TUM-VI images have a circular field of view, radius ~ 256 px at 512 px."""
    if is_tumvi(dataset):
        return 300.0 * cams[0].width / 512.0
    return None


# ------------------------------------------------------------------------------------------------
# dataset files


def read_asl_imu(mav0):
    """mav0/imu0/data.csv: t[ns], gyro xyz, accel xyz. Returns (t_ns int64, gyro N x 3, accel N x 3)."""
    f = Path(mav0) / "imu0" / "data.csv"
    t = np.loadtxt(f, delimiter=",", comments="#", usecols=0, dtype=np.int64)
    d = np.loadtxt(f, delimiter=",", comments="#")
    return t, d[:, 1:4], d[:, 4:7]


def read_asl_image_stamps(mav0, cam="cam0"):
    f = Path(mav0) / cam / "data.csv"
    return np.loadtxt(f, delimiter=",", comments="#", usecols=0, dtype=np.int64)


def parse_stamp_ns(s):
    """'1520530310.699634790' -> 1520530310699634790 (exact, no float rounding)."""
    s = s.strip()
    if "e" in s.lower():
        return int(round(float(s) * 1e9))
    if "." in s:
        a, b = s.split(".")
        b = (b + "000000000")[:9]
        sign = -1 if a.startswith("-") else 1
        return int(a) * 10**9 + sign * int(b)
    v = int(s)
    return v if v > 10**12 else v * 10**9  # seconds or nanoseconds


def read_trajectory(path, fmt="auto"):
    """Reads T_world_imu trajectories. Returns (t_ns, p N x 3, q_wxyz N x 4, v N x 3 or None).

    Formats: 'tum'   t[s] x y z qx qy qz qw (Basalt, OpenVINS, DM-VIO, SLAM Dunk exports, ...)
             'ours'  21 columns: t[s] p(3) q(w x y z) v(3) ... (sliding_marg/real outputs)
             'asl'   csv t[ns], p(3), q(w x y z) [, v(3), ...]: EuRoC ground truth, TUM-VI mocap0
    """
    path = Path(path)
    lines = [ln for ln in path.read_text().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    sep = "," if "," in lines[0] else None
    rows = [ln.split(sep) for ln in lines]
    ncol = len(rows[0])
    if fmt == "auto":
        fmt = "asl" if sep == "," else ("ours" if ncol == 21 else "tum")
    t = np.array([parse_stamp_ns(r[0]) for r in rows], dtype=np.int64)
    d = np.array([[float(v) for v in r[1:]] for r in rows])
    if fmt == "tum":
        return t, d[:, 0:3], d[:, [6, 3, 4, 5]], None
    if fmt in ("ours", "asl"):
        return t, d[:, 0:3], d[:, 3:7], (d[:, 7:10] if d.shape[1] >= 10 else None)
    raise ValueError(f"unknown trajectory format {fmt}")


def interpolate_poses(t_traj, p, q, t_query):
    """Linear / slerp interpolation; returns (p, q, valid mask)."""
    order = np.argsort(t_traj)
    t_traj, p, q = t_traj[order], p[order], q[order]
    valid = (t_query >= t_traj[0]) & (t_query <= t_traj[-1])
    j = np.clip(np.searchsorted(t_traj, t_query), 1, len(t_traj) - 1)
    a = ((t_query - t_traj[j - 1]) / np.maximum(t_traj[j] - t_traj[j - 1], 1)).clip(0, 1)
    pi = p[j - 1] + a[:, None] * (p[j] - p[j - 1])
    qi = np.array([slerp(q[j[k] - 1], q[j[k]], a[k]) for k in range(len(t_query))])
    return pi, qi, valid


def velocities(t_traj, p, q, t_query, v=None, h_ns=50_000_000):
    """Velocity at t_query: from the trajectory's own velocities if given, else central differences."""
    if v is not None:
        order = np.argsort(t_traj)
        return np.stack([np.interp(t_query, t_traj[order], v[order, i]) for i in range(3)], -1)
    lo = np.maximum(t_query - h_ns, t_traj.min())
    hi = np.minimum(t_query + h_ns, t_traj.max())
    p_lo, _, _ = interpolate_poses(t_traj, p, q, lo)
    p_hi, _, _ = interpolate_poses(t_traj, p, q, hi)
    return (p_hi - p_lo) / np.maximum(hi - lo, 1)[:, None] * 1e9


def nearest_gyro(t_imu, gyro, t_query):
    j = np.clip(np.searchsorted(t_imu, t_query), 1, len(t_imu) - 1)
    j -= (t_query - t_imu[j - 1]) < (t_imu[j] - t_query)
    return gyro[j]


# ------------------------------------------------------------------------------------------------
# writers


def write_calibration(out, cams, imu, keyframe_us, keyframe_utc_ns, device_class="Generic"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cams_json = [c.to_json() for c in cams]
    imu_json = imu.to_json()
    factory = {"DeviceClassInfo": {"DeviceClass": device_class}, "CameraCalibrations": cams_json,
               "ImuCalibrations": [imu_json]}
    (out / "factory_calibration.json").write_text(json.dumps(factory, indent=2))
    (out / "vrs_source_info.json").write_text(
        json.dumps({"camera_ids": [c.label for c in cams], "imu_ids": [imu.label]}, indent=2))
    with open(out / "online_calibration.jsonl", "w") as f:
        for us, ns in zip(keyframe_us, keyframe_utc_ns):
            f.write(json.dumps({"tracking_timestamp_us": int(us), "utc_timestamp_ns": int(ns),
                                "CameraCalibrations": cams_json, "ImuCalibrations": [imu_json]}) + "\n")


def write_imu_noise(out, noise, label="imu0"):
    if noise is not None:
        (Path(out) / "imu_noise.json").write_text(json.dumps(noise.to_json(label), indent=2))


def imu_stamp_shift_ns(t_imu_ns, shift_ms=None):
    """VIBA's preintegration holds each IMU sample constant over the period *preceding* its
    timestamp (right for Aria's averaged samples). Point samples (EuRoC, TUM-VI, most IMUs) are
    centered on that interval by shifting them by half a period (default)."""
    if shift_ms is None:
        return int(round(np.median(np.diff(t_imu_ns)) / 2))
    return int(round(shift_ms * 1e6))


def write_imu(out, t_ns, gyro, accel, label="imu0"):
    with open(Path(out) / f"imu_samples_{label}.csv", "w") as f:
        f.write("#timestamp [ns],temperature [degC],w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],"
                "w_RS_S_z [rad s^-1],a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]\n")
        for t, g, a in zip(t_ns, gyro, accel):
            f.write(f"{t},nan,{g[0]:.10f},{g[1]:.10f},{g[2]:.10f},{a[0]:.10f},{a[1]:.10f},{a[2]:.10f}\n")


def write_trajectory(out, t_us, utc_ns, p, q_wxyz, v, omega, session="session"):
    with open(Path(out) / "open_loop_trajectory.csv", "w") as f:
        f.write("tracking_timestamp_us,utc_timestamp_ns,session_uid,tx_odometry_device,ty_odometry_device,"
                "tz_odometry_device,qx_odometry_device,qy_odometry_device,qz_odometry_device,qw_odometry_device,"
                "device_linear_velocity_x_odometry,device_linear_velocity_y_odometry,"
                "device_linear_velocity_z_odometry,angular_velocity_x_device,angular_velocity_y_device,"
                "angular_velocity_z_device,gravity_x_odometry,gravity_y_odometry,gravity_z_odometry,quality_score\n")
        for k in range(len(t_us)):
            q = q_wxyz[k] / np.linalg.norm(q_wxyz[k])
            f.write(f"{t_us[k]},{utc_ns[k]},{session},"
                    f"{p[k, 0]:.10f},{p[k, 1]:.10f},{p[k, 2]:.10f},"
                    f"{q[1]:.12f},{q[2]:.12f},{q[3]:.12f},{q[0]:.12f},"
                    f"{v[k, 0]:.10f},{v[k, 1]:.10f},{v[k, 2]:.10f},"
                    f"{omega[k, 0]:.10f},{omega[k, 1]:.10f},{omega[k, 2]:.10f},0,0,-9.81,1\n")


def write_observations(out, point_id, t_us, cam_idx, uv, info_sqrt):
    """info_sqrt: scalar or per-observation sqrt information (1 / sigma_px)."""
    info_sqrt = np.broadcast_to(np.asarray(info_sqrt, float), (len(point_id),))
    with open(Path(out) / "session_observations.csv", "w") as f:
        # NOTE: capture_timestamp_ns holds microseconds (as VIBA's reader/writer expect)
        f.write("point_id,capture_timestamp_ns,camera_index,projection_base_res_x,projection_base_res_y,"
                "sqrt_h_base_res_00,sqrt_h_base_res_01,sqrt_h_base_res_10,sqrt_h_base_res_11\n")
        for k in range(len(point_id)):
            s = info_sqrt[k]
            f.write(f"{point_id[k]},{t_us[k]},{cam_idx[k]},{uv[k, 0]:.4f},{uv[k, 1]:.4f},{s:.6f},0,0,{s:.6f}\n")


def filter_tracks(point_id, kf_idx, min_keyframes=2):
    """Mask of the observations whose point is seen in at least `min_keyframes` keyframes."""
    pairs = np.unique(np.stack([point_id, kf_idx], -1), axis=0)
    ids, counts = np.unique(pairs[:, 0], return_counts=True)
    good = set(ids[counts >= min_keyframes].tolist())
    return np.array([pid in good for pid in point_id], dtype=bool)


def override_calibration(cams, calib_path, valid_radius=None):
    """Replaces intrinsics/extrinsics with those of another calibration file (same camera order),
    eg Basalt's Double Sphere calibration of TUM-VI. Returns (cameras, Imu or None, ImuNoise or None)."""
    new, imu, noise = read_calibration(calib_path)
    if len(new) < len(cams):
        raise ValueError(f"{calib_path}: {len(new)} cameras, need {len(cams)}")
    for c_old, c_new in zip(cams, new):
        c_new.label = c_old.label
        if (c_new.width, c_new.height) != (c_old.width, c_old.height):
            raise ValueError(f"{calib_path}: resolution of {c_new.label} differs from the data")
        if valid_radius is not None:
            c_new.valid_radius = valid_radius
    return new[: len(cams)], imu, noise


def sanitize_label(s):
    return re.sub(r"[^A-Za-z0-9_\-]", "_", s)
