#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Extract IMU samples and factory calibration from an Aria VRS file, as VIBA inputs.

Writes into the output folder:
  factory_calibration.json     device calibration, as stored in the VRS file
  imu_samples_<label>.csv      one file per IMU stream (eg imu-left, imu-right)

Replaces the former C++ tool `process_vrs`, so that VIBA itself does not depend on
projectaria_tools/VRS. Requires the `projectaria-tools` Python package (also required by
tools/save_observations):

  pip install projectaria-tools
  python3 process_vrs.py -i recording.vrs -o $VIBA_INPUT_DIR
"""

import argparse
from pathlib import Path

from projectaria_tools.core import calibration, data_provider
from projectaria_tools.core.sensor_data import SensorDataType

# same format as lib/motion/imu_types/ImuDataWriter.cpp
IMU_COLUMNS = [
    "#timestamp [ns]",
    "temperature [degC]",
    "w_RS_S_x [rad s^-1]",
    "w_RS_S_y [rad s^-1]",
    "w_RS_S_z [rad s^-1]",
    "a_RS_S_x [m s^-2]",
    "a_RS_S_y [m s^-2]",
    "a_RS_S_z [m s^-2]",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-i", "--in", dest="vrs", required=True, type=Path, help="VRS input")
    ap.add_argument("-o", "--out", required=True, type=Path, help="output folder (will be created)")
    args = ap.parse_args()

    provider = data_provider.create_vrs_data_provider(str(args.vrs))
    if provider is None:
        raise SystemExit(f"Unable to open: {args.vrs}")
    args.out.mkdir(parents=True, exist_ok=True)

    device_calib = provider.get_device_calibration()
    if device_calib is not None:
        (args.out / "factory_calibration.json").write_text(
            calibration.device_calibration_to_json_string(device_calib))
        print("Saved factory calibration")
    else:
        print("No device calibration in VRS file")

    for stream_id in provider.get_all_streams():
        if provider.get_sensor_data_type(stream_id) != SensorDataType.IMU:
            continue
        label = provider.get_label_from_stream_id(stream_id)
        if not label:
            print(f"IMU stream {stream_id} has no label, skipped")
            continue
        n = provider.get_num_data(stream_id)
        path = args.out / f"imu_samples_{label}.csv"
        with open(path, "w") as f:
            f.write(", ".join(IMU_COLUMNS) + "\n")
            for i in range(n):
                m = provider.get_imu_data_by_index(stream_id, i)
                g, a = m.gyro_radsec, m.accel_msec2
                f.write(f"{m.capture_timestamp_ns}, {m.temperature:.7f}, {g[0]:.7f}, {g[1]:.7f}, {g[2]:.7f}, "
                        f"{a[0]:.7f}, {a[1]:.7f}, {a[2]:.7f}\n")
        print(f"Saved {n} samples of {label} to {path}")


if __name__ == "__main__":
    main()
