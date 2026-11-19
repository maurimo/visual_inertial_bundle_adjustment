# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Reader / writer of COLMAP sparse models (text and binary), no dependency on pycolmap.

Format: https://colmap.github.io/format.html. Image poses are T_cam_world (qvec w x y z, tvec);
pixel (0.5, 0.5) is the center of the top-left pixel.
"""

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# id: (name, number of parameters)
CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
}
MODEL_IDS = {name: (i, n) for i, (name, n) in CAMERA_MODELS.items()}


@dataclass
class ColmapCamera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class ColmapImage:
    id: int
    qvec: np.ndarray  # w x y z, T_cam_world
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray  # N x 2
    point3D_ids: np.ndarray  # N, -1 if none


@dataclass
class ColmapPoint:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray


def _lines(path):
    for ln in Path(path).read_text().splitlines():
        if ln.strip() and not ln.startswith("#"):
            yield ln


def read_text(folder):
    folder = Path(folder)
    cams = {}
    for ln in _lines(folder / "cameras.txt"):
        e = ln.split()
        cams[int(e[0])] = ColmapCamera(int(e[0]), e[1], int(e[2]), int(e[3]), np.array(e[4:], float))
    images = {}
    # two lines per image, the second (points) may be empty
    raw = [ln for ln in (folder / "images.txt").read_text().splitlines() if not ln.startswith("#")]
    i = 0
    while i < len(raw):
        if not raw[i].strip():
            i += 1
            continue
        e = raw[i].split()
        pts = raw[i + 1].split() if i + 1 < len(raw) else []
        xy = np.array(pts, float).reshape(-1, 3) if pts else np.zeros((0, 3))
        images[int(e[0])] = ColmapImage(int(e[0]), np.array(e[1:5], float), np.array(e[5:8], float), int(e[8]),
                                        e[9], xy[:, :2], xy[:, 2].astype(np.int64))
        i += 2
    points = {}
    for ln in _lines(folder / "points3D.txt"):
        e = ln.split()
        tr = np.array(e[8:], np.int64).reshape(-1, 2)
        points[int(e[0])] = ColmapPoint(int(e[0]), np.array(e[1:4], float), np.array(e[4:7], int), float(e[7]),
                                        tr[:, 0], tr[:, 1])
    return cams, images, points


def read_binary(folder):
    folder = Path(folder)
    cams = {}
    with open(folder / "cameras.bin", "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            cid, mid, w, h = struct.unpack("<iiQQ", f.read(24))
            name, npar = CAMERA_MODELS[mid]
            cams[cid] = ColmapCamera(cid, name, w, h, np.array(struct.unpack(f"<{npar}d", f.read(8 * npar))))
    images = {}
    with open(folder / "images.bin", "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            iid = struct.unpack("<i", f.read(4))[0]
            q = np.array(struct.unpack("<4d", f.read(32)))
            t = np.array(struct.unpack("<3d", f.read(24)))
            cid = struct.unpack("<i", f.read(4))[0]
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (npts,) = struct.unpack("<Q", f.read(8))
            data = np.frombuffer(f.read(24 * npts), dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("id", "<i8")]))
            images[iid] = ColmapImage(iid, q, t, cid, name.decode(), np.stack([data["x"], data["y"]], -1),
                                      data["id"].astype(np.int64))
    points = {}
    with open(folder / "points3D.bin", "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            pid = struct.unpack("<Q", f.read(8))[0]
            xyz = np.array(struct.unpack("<3d", f.read(24)))
            rgb = np.array(struct.unpack("<3B", f.read(3)))
            err = struct.unpack("<d", f.read(8))[0]
            (tl,) = struct.unpack("<Q", f.read(8))
            tr = np.frombuffer(f.read(8 * tl), dtype="<i4").reshape(-1, 2)
            points[pid] = ColmapPoint(pid, xyz, rgb, err, tr[:, 0].astype(np.int64), tr[:, 1].astype(np.int64))
    return cams, images, points


def read_model(folder):
    folder = Path(folder)
    if (folder / "images.bin").exists():
        return read_binary(folder)
    if (folder / "images.txt").exists():
        return read_text(folder)
    raise FileNotFoundError(f"{folder}: no COLMAP model (images.bin / images.txt)")


def write_text(folder, cams, images, points):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for c in cams.values():
            f.write(f"{c.id} {c.model} {c.width} {c.height} " + " ".join(f"{v:.17g}" for v in c.params) + "\n")
    with open(folder / "images.txt", "w") as f:
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for im in images.values():
            f.write(f"{im.id} " + " ".join(f"{v:.17g}" for v in [*im.qvec, *im.tvec]) + f" {im.camera_id} {im.name}\n")
            f.write(" ".join(f"{x:.6f} {y:.6f} {int(p)}" for (x, y), p in zip(im.xys, im.point3D_ids)) + "\n")
    with open(folder / "points3D.txt", "w") as f:
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for p in points.values():
            f.write(f"{p.id} " + " ".join(f"{v:.17g}" for v in p.xyz) + " " + " ".join(str(int(v)) for v in p.rgb)
                    + f" {p.error:.6g} " + " ".join(f"{a} {b}" for a, b in zip(p.image_ids, p.point2D_idxs)) + "\n")
