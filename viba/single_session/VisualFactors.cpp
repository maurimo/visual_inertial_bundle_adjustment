/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/common/Enumerate.h>
#include <viba/single_session/SingleSessionAdapter.h>

#define DEFAULT_LOG_CHANNEL "ViBa::VisualFactors"
#include <logging/Log.h>

#include <fmt/format.h>
#include <viba/common/Settings.h>
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace visual_inertial_ba {

void SingleSessionAdapter::setPointHosts(const std::string& pointParam, double pointFarDist) {
  XR_CHECK(
      pointParam == "world" || pointParam == "host" || pointParam == "hostinv" ||
          pointParam == "hostmix",
      "Unknown point parameterization '{}'",
      pointParam);
  if (pointParam == "world") {
    return;
  }

  int64_t nInverseDepth = 0;
  std::vector<std::pair<int64_t, int64_t>> timeRig;
  for (size_t ptVarIndex = 0; ptVarIndex < prob_.numPointTrackVariables(); ptVarIndex++) {
    // host: the rig of the middle observation in time
    timeRig.clear();
    for (const auto& obsIndex : pointTrackObservations_[ptVarIndex]) {
      const auto& procObs = fData_.trackingObservations[obsIndex];
      timeRig.emplace_back(
          procObs.captureTimestampUs,
          findOrDie(matcher_.timestampToRigIndex, procObs.captureTimestampUs));
    }
    std::sort(timeRig.begin(), timeRig.end());
    const int64_t hostRig = timeRig[timeRig.size() / 2].second;

    bool inverseDepth = pointParam == "hostinv";
    if (pointParam == "hostmix") {
      const Vec3 ptHost =
          prob_.inertialPose(hostRig).T_bodyImu_world.value * prob_.pointTrack(ptVarIndex).value;
      inverseDepth = ptHost.norm() > pointFarDist;
    }
    prob_.pointTrack_setHost(ptVarIndex, hostRig, inverseDepth);
    nInverseDepth += inverseDepth;
  }
  if (verbosity_ != Muted) {
    XR_LOGI(
        "Points expressed in a host rig ({}): {} as inverse depth, {} as position",
        pointParam,
        nInverseDepth,
        prob_.numPointTrackVariables() - nInverseDepth);
  }
}

void SingleSessionAdapter::addVisualFactors(
    double trackingObsLossRadius,
    double trackingObsLossCutoff,
    bool optimizeDetectorBias,
    const std::string& pointParam,
    double pointFarDist) {
  // reset loss radius
  prob_.reprojErrorLoss().setSize(trackingObsLossRadius, trackingObsLossCutoff);

  if (optimizeDetectorBias) {
    throw std::runtime_error("Detector bias currently unsupported");
    // prob_.detectorBiases_init(numCameras_);
  }

  setPointHosts(pointParam, pointFarDist);

  int64_t nAdded = 0;
  for (size_t ptVarIndex = 0; ptVarIndex < prob_.numPointTrackVariables(); ptVarIndex++) {
    for (const auto& obsIndex : pointTrackObservations_[ptVarIndex]) {
      const auto& procObs = fData_.trackingObservations[obsIndex];
      const int64_t rigIndex = findOrDie(matcher_.timestampToRigIndex, procObs.captureTimestampUs);
      const int64_t cameraIndex = procObs.cameraIndex;
      auto& pointTrackVar = prob_.pointTrack(ptVarIndex);
      Vec2 projBaseRes = procObs.projectionBaseRes.cast<double>();
      Mat22 sqrtH_BaseRes = procObs.sqrtH_BaseRes.cast<double>();

      if (optimizeDetectorBias) {
        prob_.addVisualFactorWithBias(
            rigIndex,
            cameraIndex,
            pointTrackVar,
            prob_.detectorBias(cameraIndex),
            projBaseRes,
            sqrtH_BaseRes,
            prob_.reprojErrorLoss());
      } else {
        const double* timeShiftSec = nullptr;
        if (plsOn_) {
          const double exposureSec = plsExposureSec(cameraIndex, procObs.captureTimestampUs);
          const auto rsData = prob_.rigToRSData().find(rigIndex);
          if (exposureSec > 0.0 && rsData != prob_.rigToRSData().end() &&
              rsData->second.intervalHalfLengthUs_ >= plsNeededHalfIntervalUs()) {
            double* shift = prob_.addObservationTimeShift();
            const int kind =
                (!plsKinds_.empty() && plsKinds_[obsIndex] == 0) ? 0 : 1; // corner : photometric
            plsObs_.push_back(
                {shift,
                 rigIndex,
                 static_cast<int>(cameraIndex),
                 static_cast<int64_t>(ptVarIndex),
                 projBaseRes,
                 exposureSec,
                 kind});
            timeShiftSec = shift;
          }
        }
        prob_.addVisualFactor(
            rigIndex,
            cameraIndex,
            static_cast<int64_t>(ptVarIndex),
            projBaseRes,
            sqrtH_BaseRes,
            prob_.reprojErrorLoss(),
            timeShiftSec);
      }
      nAdded++;
    }
  }
  if (verbosity_ != Muted) {
    XR_LOGI("Added (tracking) visual factors: {}", nAdded);
  }
  plsUpdate();
}

// ---------------------------------------------------------------------------------------------
// Scale knots: the end of a stage

void SingleSessionAdapter::applyScaleKnots() {
  if (prob_.numScaleKnots() == 0) {
    return;
  }
  double lo = prob_.scaleKnot(0), hi = lo;
  for (int64_t i = 1; i < prob_.numScaleKnots(); i++) {
    lo = std::min(lo, prob_.scaleKnot(i));
    hi = std::max(hi, prob_.scaleKnot(i));
  }
  // points in world coordinates move with the rig of their middle observation
  std::vector<int64_t> anchorRig(prob_.numPointTrackVariables(), -1);
  for (size_t ptVarIndex = 0; ptVarIndex < anchorRig.size(); ptVarIndex++) {
    if (prob_.pointTrack_host(ptVarIndex).rigIndex >= 0 ||
        pointTrackObservations_[ptVarIndex].empty()) {
      continue;
    }
    std::vector<std::pair<int64_t, int64_t>> timeRig;
    for (const auto& obsIndex : pointTrackObservations_[ptVarIndex]) {
      const auto& procObs = fData_.trackingObservations[obsIndex];
      timeRig.emplace_back(
          procObs.captureTimestampUs,
          findOrDie(matcher_.timestampToRigIndex, procObs.captureTimestampUs));
    }
    std::sort(timeRig.begin(), timeRig.end());
    anchorRig[ptVarIndex] = timeRig[timeRig.size() / 2].second;
  }
  prob_.scaleKnots_apply(anchorRig);
  if (verbosity_ != Muted) {
    XR_LOGI(
        "Scale knots: {} knots, scale - 1 from {:+.5f} to {:+.5f}; trajectory and points divided "
        "by them",
        prob_.numScaleKnots(),
        lo - 1.0,
        hi - 1.0);
  }
}

void SingleSessionAdapter::applyYawKnots() {
  if (prob_.numYawKnots() == 0) {
    return;
  }
  prob_.yawKnots_apply();
  // what went into the trajectory since the last report (the knots may have been put in after
  // every iteration: the sums of their values; the intervals' lengths for the total turning)
  const auto applied = prob_.yawKnots_takeApplied();
  const auto lengths = prob_.yawKnotIntervalsSec();
  const double kDegPerMin = 60.0 * 180.0 / M_PI;
  double lo = applied[0][0], hi = lo, total = 0.0, maxShift = 0.0;
  std::string rates, shifts;
  for (size_t i = 0; i < applied.size(); i++) {
    lo = std::min(lo, applied[i][0]);
    hi = std::max(hi, applied[i][0]);
    total += applied[i][0] * lengths[i];
    rates += fmt::format(" {:+.4f}", applied[i][0] * kDegPerMin);
    const Vec3 u = 1e3 * applied[i].tail<3>();
    maxShift = std::max(maxShift, u.norm());
    shifts += fmt::format(" ({:+.3f} {:+.3f} {:+.3f})", u[0], u[1], u[2]);
  }
  if (verbosity_ != Muted) {
    XR_LOGI(
        "Yaw-drift knots: {} knots, rate from {:+.4f} to {:+.4f} deg/min, {:+.4f} deg over the "
        "recording; put into the trajectory. Rates [deg/min]:{}. Shift velocities up to {:.3f} "
        "mm/s [mm/s]:{}",
        prob_.numYawKnots(),
        lo * kDegPerMin,
        hi * kDegPerMin,
        total * 180.0 / M_PI,
        rates,
        maxShift,
        shifts);
  }
}

std::vector<std::pair<int64_t, int64_t>> SingleSessionAdapter::trackSpansUs() const {
  std::vector<std::pair<int64_t, int64_t>> out;
  for (const auto& observations : pointTrackObservations_) {
    int64_t firstUs = std::numeric_limits<int64_t>::max(), lastUs = 0;
    for (const auto& obsIndex : observations) {
      const int64_t t = fData_.trackingObservations[obsIndex].captureTimestampUs;
      firstUs = std::min(firstUs, t);
      lastUs = std::max(lastUs, t);
    }
    if (!observations.empty()) {
      out.emplace_back(firstUs, lastUs);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------------------------
// Parasitic light row time (InitSettings::plsScale)

namespace {
// `t_ns, exposure_s[, ...]` lines -> (us, s), sorted by time
std::vector<std::pair<int64_t, double>> readExposures(const std::filesystem::path& path) {
  std::vector<std::pair<int64_t, double>> out;
  std::ifstream in(path);
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') {
      continue;
    }
    const size_t comma = line.find(',');
    if (comma == std::string::npos) {
      continue;
    }
    try {
      const double tNs = std::stod(line.substr(0, comma));
      const double exposureSec = std::stod(line.substr(comma + 1));
      out.emplace_back(std::llround(tNs * 1e-3), exposureSec);
    } catch (const std::exception&) {
      continue; // a header
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

// Smear: the factor on the centroid time of a feature whose ghost is smeared over s = u w / l
// correlation lengths, g = eps w / te the weight of the ghost against the true image. A matcher
// puts a feature with the autocorrelation exp(-x^2 / 2 l^2) at delta = x u w with
//   x = (g / s^2) (1 - exp(-s^2 (1 - 2 x) / 2)),
// the centroid g / (2 (1 + g)) for s -> 0 and g / s^2 for s >> 1 (Newton from the centroid).
double smearAttenuation(double s, double g) {
  if (!(s > 1e-3) || !(g > 0.0)) {
    return 1.0;
  }
  const double x0 = g / (2.0 * (1.0 + g));
  const double s2 = s * s;
  double x = x0;
  for (int i = 0; i < 5; i++) {
    const double e = std::exp(-0.5 * s2 * (1.0 - 2.0 * x));
    x -= (x - (g / s2) * (1.0 - e)) / (1.0 + g * e);
  }
  return std::min(1.0, std::max(0.0, x / x0));
}
} // namespace

void SingleSessionAdapter::plsInit(const InitSettings& s) {
  plsOn_ = false;
  if (s.plsScale == 0.0 || s.plsDir.empty()) {
    return;
  }
  const std::filesystem::path dir(s.plsDir);
  plsExposure_.clear();
  for (int c = 0; c < numCameras_; c++) {
    plsExposure_.push_back(readExposures(dir / fmt::format("expo_cam{}.csv", c)));
    if (plsExposure_.back().empty()) {
      if (verbosity_ != Muted) {
        XR_LOGI(
            "Parasitic light row time: off (no exposures in {})",
            (dir / fmt::format("expo_cam{}.csv", c)).string());
      }
      plsExposure_.clear();
      return;
    }
  }
  plsOn_ = true;
  plsScale_ = s.plsScale;
  plsEps_ = s.plsEps;
  plsT0Sec_ = s.plsT0Ms * 1e-3;
  plsRowSec_ = s.plsRowUs * 1e-6;
  plsSmearPx_[0] = s.plsSmearCornerPx;
  plsSmearPx_[1] = s.plsSmearPx;

  // kinds of the observations: one line per observation, `frame, id, camera, kind`
  plsKinds_.clear();
  {
    std::ifstream in(dir / "obs_kinds.csv");
    std::string line;
    while (std::getline(in, line)) {
      const size_t comma = line.rfind(',');
      if (comma == std::string::npos || line.empty() || !std::isdigit(line[0])) {
        continue;
      }
      plsKinds_.push_back(static_cast<int8_t>(std::atoi(line.c_str() + comma + 1)));
    }
    if (plsKinds_.size() != fData_.trackingObservations.size()) {
      plsKinds_.clear();
    }
  }
  if (verbosity_ != Muted) {
    XR_LOGI(
        "Parasitic light row time: scale {}, eps 1/{:.1f}, t0 {} ms, row {} us, correlation "
        "lengths {} px (photometric) and {} px (corner tracks); kinds of the observations: {}",
        plsScale_,
        1.0 / plsEps_,
        s.plsT0Ms,
        s.plsRowUs,
        plsSmearPx_[1],
        plsSmearPx_[0],
        plsKinds_.empty() ? "unknown (all taken as photometric)" : "from obs_kinds.csv");
  }
}

double SingleSessionAdapter::plsTimeSpanSec() const {
  // the motion of the body is needed up to the largest time of the model, (t0 + rows * row) / 2
  // (+ exposure / 2 where the ghost dominates), from the frame; the rolling-shutter data cover
  // half of this span on each side, plus their own slack
  return plsOn_ ? 2.0 * (plsT0Sec_ + 480.0 * plsRowSec_) * std::max(1.0, std::abs(plsScale_))
                : 0.0;
}

// what the model needs of the motion after the frame: its largest time, (t0 + rows * row) / 2, and
// the millisecond over which the image speed is taken
int64_t SingleSessionAdapter::plsNeededHalfIntervalUs() const {
  return std::llround(
      1e6 * (0.5 * (plsT0Sec_ + 480.0 * plsRowSec_) * std::max(1.0, std::abs(plsScale_)) + 1.5e-3));
}

// exposure of the image of a camera nearest in time, within 2 ms (-1: unknown)
double SingleSessionAdapter::plsExposureSec(int cameraIndex, int64_t timestampUs) const {
  if (cameraIndex < 0 || cameraIndex >= static_cast<int>(plsExposure_.size())) {
    return -1.0;
  }
  const auto& v = plsExposure_[cameraIndex];
  auto it = std::lower_bound(v.begin(), v.end(), std::make_pair(timestampUs, -1.0));
  double best = -1.0;
  int64_t bestDt = 2000;
  for (int k = 0; k < 2; k++) {
    if (k == 1) {
      if (it == v.begin()) {
        break;
      }
      --it;
    }
    if (it != v.end() && std::llabs(it->first - timestampUs) <= bestDt) {
      bestDt = std::llabs(it->first - timestampUs);
      best = it->second;
    }
  }
  return best;
}

void SingleSessionAdapter::plsUpdate() {
  if (plsObs_.empty()) {
    return;
  }
  constexpr double kDeltaSec = 1e-3; // the image speed is taken over the first millisecond
  double sumTime = 0.0, sumShiftPx = 0.0, maxShiftPx = 0.0, sumAtt = 0.0;
  for (const auto& o : plsObs_) {
    const auto& rig = prob_.inertialPose(o.rigIndex);
    const SE3& T_bodyImu_world = rig.T_bodyImu_world.value;
    const auto& cam = prob_.cameraModel(o.rigIndex, o.cameraIndex).var.value;
    const SE3& T_Cam_BodyImu = prob_.T_Cam_BodyImu(o.rigIndex, o.cameraIndex).var.value;

    // w * (point in the body), for any way the point is expressed (VisualFactorRelative.h)
    const Vec3& v = prob_.pointTrack(o.pointIndex).value;
    const PointHost& host = prob_.pointTrack_host(o.pointIndex);
    double w = 1.0;
    Vec3 ptBody;
    if (host.rigIndex < 0) {
      ptBody = T_bodyImu_world * v;
    } else {
      Vec3 m = v;
      if (host.inverseDepth) {
        m = host.R0 * Vec3(v[0], v[1], 1.0);
        w = v[2];
      }
      if (host.rigIndex == o.rigIndex) {
        ptBody = m;
      } else {
        const SE3& T_host_world = prob_.inertialPose(host.rigIndex).T_bodyImu_world.value;
        ptBody = T_bodyImu_world.so3() * (T_host_world.so3().inverse() * (m - w * T_host_world.translation())) +
            w * T_bodyImu_world.translation();
      }
    }

    // image speed [pixels / s]: the point seen from the body one millisecond later
    double speed = 0.0;
    const auto& rsData = findOrDie(prob_.rigToRSData(), o.rigIndex);
    const SE3 T_later_mid = rsData
                                .getEstimate(
                                    kDeltaSec, rig.vel_world.value, T_bodyImu_world.inverse(), false)
                                .T_midImu_imuAtT.inverse();
    const Vec3 q0 = T_Cam_BodyImu.so3() * ptBody + w * T_Cam_BodyImu.translation();
    const Vec3 q1 = T_Cam_BodyImu.so3() * (T_later_mid.so3() * ptBody + w * T_later_mid.translation()) +
        w * T_Cam_BodyImu.translation();
    Vec2 p0, p1;
    if (q0.norm() > 0.0 && q1.norm() > 0.0 && cam.project(q0.normalized(), p0) &&
        cam.project(q1.normalized(), p1)) {
      speed = (p1 - p0).norm() / kDeltaSec;
    }

    // the model's time
    const double te = o.exposureSec;
    const double row = std::min(std::max(o.pixel.y(), 0.0), cam.imageHeight() - 1.0);
    const double wRow = plsT0Sec_ + row * plsRowSec_;
    double time = plsEps_ * wRow * (te + wRow) * 0.5 / (te + plsEps_ * wRow);
    double att = 1.0;
    const double ell = plsSmearPx_[o.kind];
    if (ell > 0.0) {
      att = smearAttenuation(speed * wRow / ell, plsEps_ * wRow / te);
    }
    time *= att;
    *o.shiftSec = plsScale_ * time;

    sumTime += time;
    sumAtt += att;
    sumShiftPx += time * speed;
    maxShiftPx = std::max(maxShiftPx, time * speed);
  }
  if (verbosity_ != Muted && plsUpdates_++ < 2) {
    XR_LOGI(
        "Parasitic light row time: {} observations with an exposure, mean time {:.3f} ms, mean "
        "attenuation by smear {:.3f}, image shift mean {:.3f} px, max {:.2f} px",
        plsObs_.size(),
        1e3 * sumTime / plsObs_.size(),
        sumAtt / plsObs_.size(),
        sumShiftPx / plsObs_.size(),
        maxShiftPx);
  }
}

} // namespace visual_inertial_ba
