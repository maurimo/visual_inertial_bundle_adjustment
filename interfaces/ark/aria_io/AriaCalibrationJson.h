/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Readers/writers for the Aria calibration JSON format (factory calibration, as extracted from
// VRS, and the per-sensor entries in MPS' online calibration). Reimplements the subset of
// projectaria_tools' loaders that is needed here, with no dependency on it.

#pragma once

#include <calibration/CameraCalibration.h>
#include <calibration/ImuCalibration.h>

#include <nlohmann/json.hpp>
#include <filesystem>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace aria_io {

/// Defaults used when a camera entry has no "ConfigData" (as in Aria factory calibrations).
struct CameraConfigData {
  int imageWidth = 0;
  int imageHeight = 0;
  double maxSolidAngle = 0.0;
  std::optional<double> maybeValidRadius;
};

/// Aria Gen1 ("Aria") / Gen2 ("Oatmeal") default camera configs by label, empty if unknown.
std::map<std::string, CameraConfigData> defaultCameraConfigs(const std::string& deviceClass);

calibration::CameraCalibration parseCameraCalibration(
    const nlohmann::json& json,
    const std::map<std::string, CameraConfigData>& defaultConfigs = {});

calibration::ImuCalibration parseImuCalibration(const nlohmann::json& json);

nlohmann::json cameraCalibrationToJson(const calibration::CameraCalibration& camCalib);

nlohmann::json imuCalibrationToJson(const calibration::ImuCalibration& imuCalib);

/// Factory (device) calibration: cameras and IMUs, by label.
struct DeviceCalibration {
  std::string deviceClass; // "Aria", "Oatmeal", or anything for non-Aria devices
  std::map<std::string, calibration::CameraCalibration> cameraCalibs;
  std::map<std::string, calibration::ImuCalibration> imuCalibs;

  // labels in alphabetical order (as projectaria_tools' DeviceCalibration)
  std::vector<std::string> getCameraLabels() const;
  std::vector<std::string> getImuLabels() const;

  std::optional<calibration::CameraCalibration> getCameraCalib(const std::string& label) const;
  std::optional<calibration::ImuCalibration> getImuCalib(const std::string& label) const;

  /// Extrinsics of camera or IMU with given label.
  std::optional<Sophus::SE3d> getT_Device_Sensor(const std::string& label) const;
};

DeviceCalibration deviceCalibrationFromJson(const std::string& jsonStr);

DeviceCalibration readDeviceCalibration(const std::filesystem::path& path);

std::string deviceCalibrationToJson(const DeviceCalibration& calib);

} // namespace aria_io
