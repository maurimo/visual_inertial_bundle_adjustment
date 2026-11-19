#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Write keyframe trajectories as `t[s] x y z qx qy qz qw` (T_world_imu, TUM-VI clock).

  export_trajectories.py viba <viba output dir> <out.txt>
      VIBA's open_loop_framerate_trajectory.csv (utc_timestamp_ns holds the TUM-VI timestamp,
      as written by slamdunk_to_viba.py; device frame = IMU frame)
  export_trajectories.py slamdunk-ba <run dir> <rec dir> <out.txt>
      SLAM Dunk's optimized keyframes (estimate/work/ba.txt)
  export_trajectories.py slamdunk-init <run dir> <rec dir> <out.txt>
      the keyframe poses SLAM Dunk's global BA starts from (estimate/work/input.bin)
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from slamdunk_to_viba import load_input_bin  # noqa: E402


def write(path, t_ns, p, q_xyzw):
    with open(path, "w") as f:
        for t, pp, q in zip(t_ns, p, q_xyzw):
            f.write(f"{t // 10**9}.{t % 10**9:09d} {pp[0]:.10f} {pp[1]:.10f} {pp[2]:.10f} "
                    f"{q[0]:.10f} {q[1]:.10f} {q[2]:.10f} {q[3]:.10f}\n")


def main():
    kind = sys.argv[1]
    if kind == "viba":
        src, out = Path(sys.argv[2]), sys.argv[3]
        d = np.genfromtxt(src / "open_loop_framerate_trajectory.csv", delimiter=",", names=True,
                          dtype=None, encoding=None)
        t_ns = np.array([int(v) for v in d["utc_timestamp_ns"]], dtype=np.int64)
        p = np.stack([d[f"t{a}_odometry_device"] for a in "xyz"], -1)
        q = np.stack([d[f"q{a}_odometry_device"] for a in "xyzw"], -1)
        write(out, t_ns, p, q)
        return
    run, rec, out = Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
    work = run / "estimate" / "work"
    offset = int(json.loads((rec / "extraction.json").read_text())["offset_ns"])
    t_ns = np.load(work / "export.npz")["t"].astype(np.int64)
    if kind == "slamdunk-ba":
        ba = np.loadtxt(work / "ba.txt")
        write(out, t_ns + offset, ba[:, 1:4], ba[:, 4:8])
    elif kind == "slamdunk-init":
        pose = load_input_bin(work / "input.bin")["pose"]  # qw qx qy qz px py pz
        write(out, t_ns + offset, pose[:, 4:7], pose[:, [1, 2, 3, 0]])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
