/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <imu_types/ImuDataReader.h>

#include <imu_types/ImuDataFormat.h>

#include <csv/CsvReader.h>

#include <limits>

namespace imu_types {
namespace {
ImuMeasurements readImpl(csv::Reader<ImuDataFormat.size()>& csv) {
  ImuMeasurements imuMeasVec;
  ImuMeasurement meas;
  std::string tempC; // read as string, it might be "nan"
  while (csv.readRow(
      meas.timestampNs,
      tempC,
      meas.gyroRadSec[0],
      meas.gyroRadSec[1],
      meas.gyroRadSec[2],
      meas.accelMSec2[0],
      meas.accelMSec2[1],
      meas.accelMSec2[2])) {
    size_t nProc;
    const double tempC_val = std::stod(tempC, &nProc);
    meas.temperatureC =
        nProc == tempC.size() ? tempC_val : std::numeric_limits<double>::quiet_NaN();
    imuMeasVec.push_back(meas);
  }

  return imuMeasVec;
}
} // namespace

ImuMeasurements ImuDataReader::read(const std::filesystem::path& path) {
  csv::Reader<ImuDataFormat.size()> csv(path, ImuDataFormat);
  return readImpl(csv);
}

ImuMeasurements ImuDataReader::read(const std::string& filename, std::istream& is) {
  csv::Reader<ImuDataFormat.size()> csv(filename, is, ImuDataFormat);
  return readImpl(csv);
}

} // namespace imu_types
