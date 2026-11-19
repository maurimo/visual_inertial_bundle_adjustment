/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <point_observation/PointObservationReader.h>

#include <csv/CsvReader.h>

#include <point_observation/PointObservationFormat.h>

namespace visual_inertial_ba {
namespace {
PointObservations readImpl(csv::Reader<PointObservationColumns.size()>& csv) {
  PointObservations trackingObservations;

  PointObservation tObs;
  while (csv.readRow(
      tObs.pointId,
      tObs.captureTimestampUs,
      tObs.cameraIndex,
      tObs.projectionBaseRes.x(),
      tObs.projectionBaseRes.y(),
      tObs.sqrtH_BaseRes(0, 0),
      tObs.sqrtH_BaseRes(0, 1),
      tObs.sqrtH_BaseRes(1, 0),
      tObs.sqrtH_BaseRes(1, 1))) {
    trackingObservations.push_back(tObs);
  }

  return trackingObservations;
}
} // namespace

PointObservations PointObservationReader::read(const std::filesystem::path& path) {
  csv::Reader<PointObservationColumns.size()> csv(path, PointObservationColumns);
  return readImpl(csv);
}
PointObservations PointObservationReader::read(const std::string& filename, std::istream& is) {
  csv::Reader<PointObservationColumns.size()> csv(filename, is, PointObservationColumns);
  return readImpl(csv);
}
} // namespace visual_inertial_ba
