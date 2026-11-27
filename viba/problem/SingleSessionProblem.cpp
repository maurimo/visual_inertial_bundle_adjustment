/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// @lint-ignore-every CLANGTIDY facebook-hte-LocalUncheckedArrayBounds
#include <cstdlib>
#include <viba/common/Enumerate.h>
#include <viba/problem/Histograms.h>
#include <viba/problem/SingleSessionProblem.h>

#include <logging/Checks.h>
#define DEFAULT_LOG_CHANNEL "ViBa::SingleSessionProblem"
#include <logging/Log.h>

#include <cmath>

namespace visual_inertial_ba {

SingleSessionProblem::SingleSessionProblem(small_thing::Optimizer& opt, Verbosity verbosity)
    : verbosity_(verbosity), opt_(opt) {}

void SingleSessionProblem::setAllRigsConstant(bool constPose, bool constLinVel, bool constAngVel) {
  if (verbosity_ != Muted) {
    XR_LOGI(
        "Setting {} inertial rigs constness: (constPose: {}, constLinVel: {}, constAngVel: {})",
        inertialPoses_.size(),
        constPose,
        constLinVel,
        constAngVel);
  }
  for (auto& [_, inertialPose] : inertialPoses_) {
    XR_CHECK(!inertialPose.T_bodyImu_world.isRegistered());
    XR_CHECK(!inertialPose.vel_world.isRegistered());
    XR_CHECK(!inertialPose.omega.isRegistered());
    inertialPose.T_bodyImu_world.setConstant(constPose);
    inertialPose.vel_world.setConstant(constLinVel);
    inertialPose.omega.setConstant(constAngVel);
  }
}

void SingleSessionProblem::registerPointVariables() {
  for (auto& ptVar : pointTracks_) {
    opt_.registerVariable(ptVar);
  }
  // loop-closing points and their virtual points (none without loop closure)
  for (auto& ptVar : loopPoints_) {
    opt_.registerVariable(ptVar);
  }
  for (auto& ptVar : loopVirtualPoints_) {
    opt_.registerVariable(ptVar);
  }
}

std::optional<int64_t> SingleSessionProblem::firstRigIndex() const {
  std::optional<int64_t> firstRigIndex;
  for (const auto& [rigIndex, _] : inertialPoses_) {
    if (!firstRigIndex.has_value() || (rigIndex < *firstRigIndex)) {
      firstRigIndex = rigIndex;
    }
  }
  return firstRigIndex;
}

std::vector<int64_t> SingleSessionProblem::sortedRigIndices() const {
  std::vector<int64_t> rigIndices;
  for (const auto& [rigIndex, _] : inertialPoses_) {
    rigIndices.push_back(rigIndex);
  }
  std::sort(rigIndices.begin(), rigIndices.end());
  return rigIndices;
}

void SingleSessionProblem::computeCovariances() {
  const auto maybeFirstRigIndex = firstRigIndex();
  if (!maybeFirstRigIndex.has_value()) {
    return;
  }

  // make sure position and yaw of first rig are constained
  XR_CHECK_EQ(
      clearPosePriors(), 0, "Computation of covariances: problem should have no pose priors");
  constrainPositionAndYaw(*maybeFirstRigIndex);

  // indices for computation of covariance, need optimizer indices and rig index for reference
  std::unordered_set<int64_t> nonRigCovVarIndices;
  for (const auto& imuVar : imuCalibs_) {
    XR_CHECK_GE(imuVar.var.index, 0); // TODO: allow set to constant
    nonRigCovVarIndices.insert(imuVar.var.index);
  }
  for (const auto& extrVar : Ts_Imu_BodyImu_) {
    XR_CHECK_GE(extrVar.var.index, 0); // TODO: allow set to constant
    nonRigCovVarIndices.insert(extrVar.var.index);
  }
  for (const auto& intrVar : cameraModels_) {
    XR_CHECK_GE(intrVar.var.index, 0); // TODO: allow set to constant
    nonRigCovVarIndices.insert(intrVar.var.index);
  }
  for (const auto& extrVar : Ts_Cam_BodyImu_) {
    XR_CHECK_GE(extrVar.var.index, 0); // TODO: allow set to constant
    nonRigCovVarIndices.insert(extrVar.var.index);
  }

  bool estimatingOmega = findOrDie(inertialPoses_, *maybeFirstRigIndex).omega.isRegistered();

  std::vector<std::vector<int64_t>> covVarIndices;
  for (const auto& [rigIndex, var] : inertialPoses_) {
    XR_CHECK(var.T_bodyImu_world.isRegistered());
    XR_CHECK(var.vel_world.isRegistered());
    XR_CHECK_EQ(estimatingOmega, var.omega.isRegistered());

    // add all variables for the rig (pose, velocity, etc)
    covVarIndices.push_back({var.T_bodyImu_world.index, var.vel_world.index});
    if (estimatingOmega) {
      covVarIndices.back().push_back(var.omega.index);
    }
  }

  // add the non-rig variables, made unique
  for (int64_t index : nonRigCovVarIndices) {
    covVarIndices.push_back({index});
  }

  if (verbosity_ != Muted) {
    XR_LOGI(
        "Computing covs... ({} variables: {} rigs, {} imu calibs, {} imu extr, {} cam intr, {} cam extr)",
        covVarIndices.size(),
        inertialPoses_.size(),
        imuCalibs_.size(),
        Ts_Imu_BodyImu_.size(),
        cameraModels_.size(),
        Ts_Cam_BodyImu_.size());
  }
  auto covAndOffsets = opt_.computeJointCovariances({.damping = 0}, covVarIndices);
  XR_CHECK_EQ(covAndOffsets.size(), covVarIndices.size());

  // remove the pose prior we added, to leave the problem in a "clean" state
  XR_CHECK_EQ(clearPosePriors(), 1);

  if (verbosity_ != Muted) {
    XR_LOGI("Done!");
  }
  for (auto [i, varIndices] : enumerate(covVarIndices)) {
    variableCovariances_[varIndices[0]] = std::move(covAndOffsets[i].first);
  }
}

int64_t SingleSessionProblem::numRigVariables() const {
  return inertialPoses_.size();
}

int64_t SingleSessionProblem::numCameraModelVariables() const {
  return cameraModels_.size();
}

int64_t SingleSessionProblem::numExtrinsicsVariables() const {
  return Ts_Cam_BodyImu_.size() + Ts_Imu_BodyImu_.size();
}

int64_t SingleSessionProblem::numCameraExtrinsicsVariables() const {
  return Ts_Cam_BodyImu_.size();
}

int64_t SingleSessionProblem::numImuExtrinsicsVariables() const {
  return Ts_Imu_BodyImu_.size();
}

int64_t SingleSessionProblem::numImuCalibVariables() const {
  return imuCalibs_.size();
}

int64_t SingleSessionProblem::numPointTrackVariables() const {
  return pointTracks_.size();
}

const std::unordered_map<int64_t, Eigen::MatrixXd>& SingleSessionProblem::variableCovariances()
    const {
  return variableCovariances_;
}

const std::unordered_map<int64_t, InertialPoseVariables>& SingleSessionProblem::inertialPoses()
    const {
  return inertialPoses_;
}

bool SingleSessionProblem::inertialPose_exists(int64_t rigIndex) {
  return inertialPoses_.count(rigIndex) > 0;
}

void SingleSessionProblem::inertialPose_set(
    int64_t rigIndex,
    const SE3& T_bodyImu_world,
    const Vec3& vel_world,
    const Vec3& omega,
    int64_t timestampUs) {
  inertialPoses_[rigIndex] = {
      .T_bodyImu_world = T_bodyImu_world,
      .vel_world = vel_world,
      .omega = omega,
      .timestampUs = timestampUs,
  };
}

InertialPoseVariables& SingleSessionProblem::inertialPose(int64_t rigIndex) {
  return findOrDie(inertialPoses_, rigIndex);
}

const InertialPoseVariables& SingleSessionProblem::inertialPose(int64_t rigIndex) const {
  return findOrDie(inertialPoses_, rigIndex);
}

InertialPoseVariables* SingleSessionProblem::inertialPose_find(int64_t rigIndex) {
  const auto it = inertialPoses_.find(rigIndex);
  return it != inertialPoses_.end() ? &it->second : nullptr;
}

const InertialPoseVariables* SingleSessionProblem::inertialPose_find(int64_t rigIndex) const {
  const auto it = inertialPoses_.find(rigIndex);
  return it != inertialPoses_.end() ? &it->second : nullptr;
}

void SingleSessionProblem::gravityWorld_initNew(const GravityData& gravityWorld, bool constant) {
  gravityWorld_ = std::make_shared<GravityVariable>(gravityWorld);
  gravityWorld_->setConstant(constant);
}

void SingleSessionProblem::gravityWorld_setRef(
    const std::shared_ptr<GravityVariable>& gravityWorldVar) {
  gravityWorld_ = gravityWorldVar;
}

GravityVariable& SingleSessionProblem::gravityWorld() {
  XR_CHECK(gravityWorld_);
  return *gravityWorld_;
}

const GravityVariable& SingleSessionProblem::gravityWorld() const {
  XR_CHECK(gravityWorld_);
  return *gravityWorld_;
}

void SingleSessionProblem::gravityKnots_init(
    int64_t startTimestampUs,
    int64_t endTimestampUs,
    int numKnots,
    const Vec3& priorDiagSqrtH,
    const Vec3& randomWalkDiagSqrtH) {
  XR_CHECK(gravityKnots_.empty());
  XR_CHECK_GE(numKnots, 2);
  XR_CHECK_GT(endTimestampUs, startTimestampUs);
  const Vec3 nominal = gravityWorld().value.vec;
  gravityKnotsStartUs_ = startTimestampUs;
  gravityKnotsSpacingUs_ = double(endTimestampUs - startTimestampUs) / (numKnots - 1);
  // (the optimizer keeps pointers to the variables: all are created before any factor)
  gravityKnots_.reserve(numKnots);
  for (int i = 0; i < numKnots; i++) {
    gravityKnots_.emplace_back(nominal);
  }
  for (int i = 0; i < numKnots; i++) {
    opt_.addFactor(
        [nominal, priorDiagSqrtH](const Vec3& g, Ref<Mat33>&& gJacobian) -> Vec3 {
          if (!isNull(gJacobian)) {
            gJacobian = priorDiagSqrtH.asDiagonal();
          }
          return priorDiagSqrtH.cwiseProduct(g - nominal);
        },
        gravityKnots_[i]);
    if (i + 1 < numKnots) {
      opt_.addFactor(
          [randomWalkDiagSqrtH](
              const Vec3& gPrev,
              const Vec3& gNext,
              Ref<Mat33>&& gPrevJacobian,
              Ref<Mat33>&& gNextJacobian) -> Vec3 {
            if (!isNull(gPrevJacobian)) {
              gPrevJacobian = -Mat33(randomWalkDiagSqrtH.asDiagonal());
            }
            if (!isNull(gNextJacobian)) {
              gNextJacobian = randomWalkDiagSqrtH.asDiagonal();
            }
            return randomWalkDiagSqrtH.cwiseProduct(gNext - gPrev);
          },
          gravityKnots_[i],
          gravityKnots_[i + 1]);
    }
  }
}

int64_t SingleSessionProblem::numGravityKnots() const {
  return gravityKnots_.size();
}

int64_t SingleSessionProblem::gravityKnotTimestampUs(int64_t knotIndex) const {
  return gravityKnotsStartUs_ + std::llround(knotIndex * gravityKnotsSpacingUs_);
}

const Vec3& SingleSessionProblem::gravityKnot(int64_t knotIndex) const {
  return gravityKnots_.at(knotIndex).value;
}

std::pair<int64_t, double> SingleSessionProblem::gravityKnotAt(int64_t timestampUs) const {
  XR_CHECK_GE(gravityKnots_.size(), 2);
  const double x = (timestampUs - gravityKnotsStartUs_) / gravityKnotsSpacingUs_;
  const int64_t last = gravityKnots_.size() - 2;
  const int64_t k = std::min(std::max<int64_t>(int64_t(std::floor(x)), 0), last);
  return {k, std::min(std::max(x - k, 0.0), 1.0)};
}

void SingleSessionProblem::scaleKnots_init(
    int64_t startTimestampUs,
    int64_t endTimestampUs,
    int numKnots,
    double priorSqrtH,
    double randomWalkSqrtH) {
  XR_CHECK(scaleKnots_.empty());
  XR_CHECK_GE(numKnots, 2);
  XR_CHECK_GT(endTimestampUs, startTimestampUs);
  scaleKnotsStartUs_ = startTimestampUs;
  scaleKnotsSpacingUs_ = double(endTimestampUs - startTimestampUs) / (numKnots - 1);
  // (the optimizer keeps pointers to the variables: all are created before any factor)
  scaleKnots_.reserve(numKnots);
  for (int i = 0; i < numKnots; i++) {
    scaleKnots_.emplace_back(Vec<1>::Ones());
  }
  using Mat11 = Mat<1, 1>;
  for (int i = 0; i < numKnots; i++) {
    opt_.addFactor(
        [priorSqrtH](const Vec<1>& s, Ref<Mat11>&& sJacobian) -> Vec<1> {
          if (!isNull(sJacobian)) {
            sJacobian(0, 0) = priorSqrtH;
          }
          return Vec<1>(priorSqrtH * (s[0] - 1.0));
        },
        scaleKnots_[i]);
    if (i + 1 < numKnots) {
      opt_.addFactor(
          [randomWalkSqrtH](
              const Vec<1>& sPrev,
              const Vec<1>& sNext,
              Ref<Mat11>&& sPrevJacobian,
              Ref<Mat11>&& sNextJacobian) -> Vec<1> {
            if (!isNull(sPrevJacobian)) {
              sPrevJacobian(0, 0) = -randomWalkSqrtH;
            }
            if (!isNull(sNextJacobian)) {
              sNextJacobian(0, 0) = randomWalkSqrtH;
            }
            return Vec<1>(randomWalkSqrtH * (sNext[0] - sPrev[0]));
          },
          scaleKnots_[i],
          scaleKnots_[i + 1]);
    }
  }
}

int64_t SingleSessionProblem::numScaleKnots() const {
  return scaleKnots_.size();
}

int64_t SingleSessionProblem::scaleKnotTimestampUs(int64_t knotIndex) const {
  return scaleKnotsStartUs_ + std::llround(knotIndex * scaleKnotsSpacingUs_);
}

double SingleSessionProblem::scaleKnot(int64_t knotIndex) const {
  return scaleKnots_.at(knotIndex).value[0];
}

double SingleSessionProblem::scaleKnots_valueAt(int64_t timestampUs) const {
  if (scaleKnots_.empty()) {
    return 1.0;
  }
  const auto [knot, alpha] = scaleKnotAt(timestampUs);
  return (1.0 - alpha) * scaleKnots_[knot].value[0] + alpha * scaleKnots_[knot + 1].value[0];
}

void SingleSessionProblem::scaleKnots_apply(const std::vector<int64_t>& worldPointAnchorRig) {
  if (scaleKnots_.empty()) {
    return;
  }
  // positions of the IMU, rebuilt from the steps
  std::unordered_map<int64_t, Vec3> oldPos, newPos;
  Vec3 prevOld = Vec3::Zero(), prevNew = Vec3::Zero();
  int64_t prevTimestampUs = -1;
  const auto rigs = sortedRigIndices();
  for (const int64_t j : rigs) {
    const auto& rig = findOrDie(inertialPoses_, j);
    const Vec3 pos = rig.T_bodyImu_world.value.inverse().translation();
    Vec3 pos2 = pos;
    if (prevTimestampUs >= 0) {
      pos2 = prevNew +
          (pos - prevOld) / scaleKnots_valueAt((prevTimestampUs + rig.timestampUs) / 2);
    }
    oldPos[j] = pos;
    newPos[j] = pos2;
    prevOld = pos;
    prevNew = pos2;
    prevTimestampUs = rig.timestampUs;
  }
  // points
  for (size_t i = 0; i < pointTracks_.size(); i++) {
    Vec3& v = pointTracks_[i].value;
    const PointHost& host = pointHosts_[i];
    if (host.rigIndex >= 0) {
      const double s = scaleKnots_valueAt(findOrDie(inertialPoses_, host.rigIndex).timestampUs);
      if (host.inverseDepth) {
        v[2] *= s;
      } else {
        v /= s;
      }
    } else if (i < worldPointAnchorRig.size() && oldPos.count(worldPointAnchorRig[i]) > 0) {
      const int64_t j = worldPointAnchorRig[i];
      const double s = scaleKnots_valueAt(findOrDie(inertialPoses_, j).timestampUs);
      v = newPos[j] + (v - oldPos[j]) / s;
    }
  }
  // poses and velocities
  for (const int64_t j : rigs) {
    auto& rig = findOrDie(inertialPoses_, j);
    const SO3 R_bodyImu_world = rig.T_bodyImu_world.value.so3();
    rig.T_bodyImu_world.value = SE3(R_bodyImu_world, -(R_bodyImu_world * newPos[j]));
    rig.vel_world.value /= scaleKnots_valueAt(rig.timestampUs);
  }
  for (auto& knot : scaleKnots_) {
    knot.value[0] = 1.0;
  }
}

std::pair<int64_t, double> SingleSessionProblem::scaleKnotAt(int64_t timestampUs) const {
  XR_CHECK_GE(scaleKnots_.size(), 2);
  const double x = (timestampUs - scaleKnotsStartUs_) / scaleKnotsSpacingUs_;
  const int64_t last = scaleKnots_.size() - 2;
  const int64_t k = std::min(std::max<int64_t>(int64_t(std::floor(x)), 0), last);
  return {k, std::min(std::max(x - k, 0.0), 1.0)};
}

std::vector<double> SingleSessionProblem::yawKnotIntervalsSec() const {
  std::vector<double> out;
  for (size_t k = 0; k + 1 < yawPath_.knotTimeSec.size(); k++) {
    out.push_back(yawPath_.knotTimeSec[k + 1] - yawPath_.knotTimeSec[k]);
  }
  return out;
}

void SingleSessionProblem::yawKnots_init(
    double minKnotSec,
    const std::vector<std::pair<int64_t, int64_t>>& trackSpansUs,
    double priorSqrtH,
    double randomWalkSqrtH,
    double shiftSqrtH) {
  XR_CHECK(yawKnots_.empty());
  XR_CHECK_GT(minKnotSec, 0.0);
  const auto rigs = sortedRigIndices();
  XR_CHECK_GE(rigs.size(), 2);
  yawKnotsStartUs_ = findOrDie(inertialPoses_, rigs.front()).timestampUs;
  const double lengthSec =
      (findOrDie(inertialPoses_, rigs.back()).timestampUs - yawKnotsStartUs_) * 1e-6;
  yawPath_ = YawDriftPath();
  yawPath_.shift = shiftSqrtH > 0.0;
  // the intervals: [0, minKnotSec), then each next one ends minKnotSec later or just after the
  // last observation of the tracks that start in the current one, whichever is later
  {
    std::vector<std::pair<double, double>> spans; // (first, last) [s], sorted by first
    for (const auto& [firstUs, lastUs] : trackSpansUs) {
      spans.emplace_back((firstUs - yawKnotsStartUs_) * 1e-6, (lastUs - yawKnotsStartUs_) * 1e-6);
    }
    std::sort(spans.begin(), spans.end());
    const double kAfterSec = 1e-3;
    auto& times = yawPath_.knotTimeSec;
    times.push_back(0.0);
    size_t next = 0;
    double end = std::min(minKnotSec, lengthSec);
    while (true) {
      // a rest shorter than the minimum goes with this interval
      if (lengthSec - end < minKnotSec) {
        end = lengthSec;
      }
      double lastOfTracks = 0.0;
      for (; next < spans.size() && spans[next].first < end; next++) {
        lastOfTracks = std::max(lastOfTracks, spans[next].second);
      }
      times.push_back(end);
      if (end >= lengthSec) {
        break;
      }
      end = std::max(end + minKnotSec, lastOfTracks + kAfterSec);
    }
    if (times.size() == 2) {
      times = {0.0, 0.5 * lengthSec, lengthSec}; // (two knots at least)
    }
  }
  const int numKnots = yawPath_.numKnots();
  yawRigOrder_.clear();
  for (const int64_t j : rigs) {
    const auto& rig = findOrDie(inertialPoses_, j);
    yawRigOrder_[j] = int(yawPath_.timeSec.size());
    yawPath_.timeSec.push_back((rig.timestampUs - yawKnotsStartUs_) * 1e-6);
    yawPath_.refPos.push_back(rig.T_bodyImu_world.value.inverse().translation());
  }
  // (the optimizer keeps pointers to the variables: all are created before any factor)
  yawKnots_.reserve(numKnots);
  // (test: VIBA_YAW_KNOT_INIT=<rad/s> starts the knots away from 0, alternating in sign, so that
  // --verify-jacobians checks the factors with a drift)
  const double testInit =
      std::getenv("VIBA_YAW_KNOT_INIT") ? std::atof(std::getenv("VIBA_YAW_KNOT_INIT")) : 0.0;
  for (int i = 0; i < numKnots; i++) {
    const double sign = i % 2 == 0 ? 1.0 : -0.6;
    Vec<4> init(testInit * sign, 0.0, 0.0, 0.0);
    if (yawPath_.shift) {
      init.tail<3>() = 20.0 * testInit * sign * Vec3(1.0, -0.5, 0.25);
    }
    yawKnots_.emplace_back(init);
  }
  using Mat44 = Mat<4, 4>;
  using Mat14 = Mat<1, 4>;
  // (without the shift the velocities have no other factor: they stay 0)
  const double shiftPrior = yawPath_.shift ? shiftSqrtH : 1.0;
  const Vec<4> sqrtH(priorSqrtH, shiftPrior, shiftPrior, shiftPrior);
  for (int i = 0; i < numKnots; i++) {
    opt_.addFactor(
        [sqrtH](const Vec<4>& w, Ref<Mat44>&& wJacobian) -> Vec<4> {
          if (!isNull(wJacobian)) {
            wJacobian = Mat44(sqrtH.asDiagonal());
          }
          return Vec<4>(sqrtH.cwiseProduct(w));
        },
        yawKnots_[i]);
    if (i + 1 < numKnots && randomWalkSqrtH > 0.0) {
      opt_.addFactor(
          [randomWalkSqrtH](
              const Vec<4>& wPrev,
              const Vec<4>& wNext,
              Ref<Mat14>&& wPrevJacobian,
              Ref<Mat14>&& wNextJacobian) -> Vec<1> {
            if (!isNull(wPrevJacobian)) {
              wPrevJacobian.setZero();
              wPrevJacobian(0, 0) = -randomWalkSqrtH;
            }
            if (!isNull(wNextJacobian)) {
              wNextJacobian.setZero();
              wNextJacobian(0, 0) = randomWalkSqrtH;
            }
            return Vec<1>(randomWalkSqrtH * (wNext[0] - wPrev[0]));
          },
          yawKnots_[i],
          yawKnots_[i + 1]);
    }
  }
}

int64_t SingleSessionProblem::numYawKnots() const {
  return yawKnots_.size();
}

int64_t SingleSessionProblem::yawKnotTimestampUs(int64_t knotIndex) const {
  return yawKnotsStartUs_ + std::llround(yawPath_.knotTimeSec.at(knotIndex) * 1e6);
}

double SingleSessionProblem::yawKnot(int64_t knotIndex) const {
  return yawKnots_.at(knotIndex).value[0];
}

Vec3 SingleSessionProblem::yawKnotShift(int64_t knotIndex) const {
  return yawKnots_.at(knotIndex).value.tail<3>();
}

std::vector<SingleSessionProblem::TrueRigState> SingleSessionProblem::yawKnots_trueStates() const {
  std::vector<TrueRigState> out;
  const bool drift = !yawKnots_.empty() && yawPath_.active;
  const auto knot = [&](int k) { return yawKnots_[k].value[0]; };
  const auto knotVel = [&](int k) {
    return yawPath_.shift ? Vec3(yawKnots_[k].value.tail<3>()) : Vec3(Vec3::Zero());
  };
  double prevPsi = 0.0;
  Vec3 q = Vec3::Zero();
  int i = 0;
  for (const int64_t j : sortedRigIndices()) {
    const auto& rig = findOrDie(inertialPoses_, j);
    TrueRigState s{j, rig.timestampUs, rig.T_bodyImu_world.value, rig.vel_world.value, 0.0};
    if (drift) {
      XR_CHECK_EQ(findOrDie(yawRigOrder_, j), i);
      s.psi = yawPath_.psiAt(yawPath_.timeSec[i], knot);
      q = i == 0
          ? yawPath_.refPos[0]
          : Vec3(q + SO3::rotZ(0.5 * (prevPsi + s.psi)) *
                    (yawPath_.refPos[i] - yawPath_.refPos[i - 1] +
                     yawPath_.shiftBetween(yawPath_.timeSec[i - 1], yawPath_.timeSec[i], knotVel)));
      const SO3 Rz = SO3::rotZ(s.psi);
      const Vec3 pos = rig.T_bodyImu_world.value.inverse().translation();
      const Vec3 truePos = q + Rz * (pos - yawPath_.refPos[i]);
      const SO3 R_bodyImu_world = rig.T_bodyImu_world.value.so3() * Rz.inverse();
      s.T_bodyImu_world = SE3(R_bodyImu_world, -(R_bodyImu_world * truePos));
      s.vel_world = Rz * rig.vel_world.value;
      prevPsi = s.psi;
    }
    out.push_back(s);
    i++;
  }
  return out;
}

double SingleSessionProblem::yawKnots_apply() {
  if (yawKnots_.empty()) {
    return 0.0;
  }
  double psi = 0.0;
  for (const auto& s : yawKnots_trueStates()) {
    auto& rig = findOrDie(inertialPoses_, s.rigIndex);
    rig.T_bodyImu_world.value = s.T_bodyImu_world;
    rig.vel_world.value = s.vel_world;
    psi = s.psi;
  }
  yawKnotsApplied_.resize(yawKnots_.size(), Vec<4>::Zero());
  for (size_t k = 0; k < yawKnots_.size(); k++) {
    yawKnotsApplied_[k] += yawKnots_[k].value;
    yawKnots_[k].value.setZero();
  }
  yawKnots_refreeze();
  return psi;
}

std::vector<Vec<4>> SingleSessionProblem::yawKnots_takeApplied() {
  std::vector<Vec<4>> out(yawKnots_.size(), Vec<4>::Zero());
  out.swap(yawKnotsApplied_);
  out.resize(yawKnots_.size(), Vec<4>::Zero());
  return out;
}

void SingleSessionProblem::yawKnots_refreeze() {
  if (yawKnots_.empty()) {
    return;
  }
  for (const auto& knot : yawKnots_) {
    XR_CHECK(knot.value.isZero(0.0));
  }
  for (const auto& [j, i] : yawRigOrder_) {
    yawPath_.refPos[i] = findOrDie(inertialPoses_, j).T_bodyImu_world.value.inverse().translation();
  }
}

void SingleSessionProblem::yawKnots_gaugeTest(double rate) {
  XR_CHECK(!yawKnots_.empty() && yawPath_.active);
  // the true states now, then the knots, then the variables that give the same true states
  const auto before = yawKnots_trueStates();
  for (size_t k = 0; k < yawKnots_.size(); k++) {
    const double f = k % 3 == 0 ? 1.0 : k % 3 == 1 ? -0.7 : 0.4;
    yawKnots_[k].value[0] = rate * f;
    if (yawPath_.shift) {
      yawKnots_[k].value.tail<3>() = 30.0 * rate * f * Vec3(1.0, -0.5, 0.25);
    }
  }
  const auto knot = [&](int k) { return yawKnots_[k].value[0]; };
  const auto knotVel = [&](int k) {
    return yawPath_.shift ? Vec3(yawKnots_[k].value.tail<3>()) : Vec3(Vec3::Zero());
  };
  double prevPsi = 0.0;
  Vec3 q = Vec3::Zero();
  for (size_t i = 0; i < before.size(); i++) {
    const double psi = yawPath_.psiAt(yawPath_.timeSec[i], knot);
    q = i == 0
        ? yawPath_.refPos[0]
        : Vec3(q + SO3::rotZ(0.5 * (prevPsi + psi)) *
                  (yawPath_.refPos[i] - yawPath_.refPos[i - 1] +
                   yawPath_.shiftBetween(yawPath_.timeSec[i - 1], yawPath_.timeSec[i], knotVel)));
    prevPsi = psi;
    const SO3 Rz = SO3::rotZ(psi);
    const Vec3 truePos = before[i].T_bodyImu_world.inverse().translation();
    const Vec3 pos = yawPath_.refPos[i] + Rz.inverse() * (truePos - q);
    const SO3 R_bodyImu_world = before[i].T_bodyImu_world.so3() * Rz;
    auto& rig = findOrDie(inertialPoses_, before[i].rigIndex);
    rig.T_bodyImu_world.value = SE3(R_bodyImu_world, -(R_bodyImu_world * pos));
    rig.vel_world.value = Rz.inverse() * before[i].vel_world;
  }
}

Vec3 SingleSessionProblem::gravityAt(int64_t timestampUs) const {
  if (gravityKnots_.empty()) {
    return gravityWorld().value.vec;
  }
  const auto [k, alpha] = gravityKnotAt(timestampUs);
  return (1.0 - alpha) * gravityKnots_[k].value + alpha * gravityKnots_[k + 1].value;
}

int64_t SingleSessionProblem::cameraModel_addNew(
    CameraModelParam&& data,
    int64_t averageTimestampUs,
    int64_t prevVarIndex,
    int sensorIndex,
    bool constant) {
  cameraModels_.push_back(
      CameraModelVariable{
          .var = std::move(data),
          .averageTimestampUs = averageTimestampUs,
          .prevVarIndex = prevVarIndex,
          .sensorIndex = sensorIndex,
      });
  cameraModels_.back().var.setConstant(constant);
  return cameraModels_.size() - 1;
}

CameraModelVariable& SingleSessionProblem::cameraModel_at(int64_t index) {
  return cameraModels_[index];
}

const CameraModelVariable& SingleSessionProblem::cameraModel_at(int64_t index) const {
  return cameraModels_[index];
}

const SingleSessionProblem::RigSensorToIndex& SingleSessionProblem::rigCamToModelIndex() const {
  return rigCamToModelIndex_;
}

void SingleSessionProblem::cameraModel_setRef(int64_t rigIndex, int s, int index) {
  rigCamToModelIndex_[{rigIndex, s}] = index;
}

CameraModelVariable& SingleSessionProblem::cameraModel(int64_t rigIndex, int s) {
  return cameraModels_[findOrDie(rigCamToModelIndex_, {rigIndex, s})];
}

const CameraModelVariable& SingleSessionProblem::cameraModel(int64_t rigIndex, int s) const {
  return cameraModels_[findOrDie(rigCamToModelIndex_, {rigIndex, s})];
}

CameraModelVariable* SingleSessionProblem::cameraModel_find(int64_t rigIndex, int s) {
  const auto it = rigCamToModelIndex_.find({rigIndex, s});
  return it != rigCamToModelIndex_.end() ? &cameraModels_[it->second] : nullptr;
}

const CameraModelVariable* SingleSessionProblem::cameraModel_find(int64_t rigIndex, int s) const {
  const auto it = rigCamToModelIndex_.find({rigIndex, s});
  return it != rigCamToModelIndex_.end() ? &cameraModels_[it->second] : nullptr;
}

int64_t SingleSessionProblem::T_Cam_BodyImu_addNew(
    const SE3& data,
    int64_t averageTimestampUs,
    int64_t prevVarIndex,
    int sensorIndex,
    bool constant) {
  Ts_Cam_BodyImu_.push_back(
      ExtrinsicsVariable{
          .var = data,
          .averageTimestampUs = averageTimestampUs,
          .prevVarIndex = prevVarIndex,
          .sensorIndex = sensorIndex,
      });
  Ts_Cam_BodyImu_.back().var.setConstant(constant);
  return Ts_Cam_BodyImu_.size() - 1;
}

ExtrinsicsVariable& SingleSessionProblem::T_Cam_BodyImu_at(int64_t index) {
  return Ts_Cam_BodyImu_[index];
}

const ExtrinsicsVariable& SingleSessionProblem::T_Cam_BodyImu_at(int64_t index) const {
  return Ts_Cam_BodyImu_[index];
}

const SingleSessionProblem::RigSensorToIndex& SingleSessionProblem::rigCamToExtrIndex() const {
  return rigCamToExtrIndex_;
}

void SingleSessionProblem::T_Cam_BodyImu_setRef(int64_t rigIndex, int s, int index) {
  rigCamToExtrIndex_[{rigIndex, s}] = index;
}

ExtrinsicsVariable& SingleSessionProblem::T_Cam_BodyImu(int64_t rigIndex, int s) {
  return Ts_Cam_BodyImu_[findOrDie(rigCamToExtrIndex_, {rigIndex, s})];
}

const ExtrinsicsVariable& SingleSessionProblem::T_Cam_BodyImu(int64_t rigIndex, int s) const {
  return Ts_Cam_BodyImu_[findOrDie(rigCamToExtrIndex_, {rigIndex, s})];
}

ExtrinsicsVariable* SingleSessionProblem::T_Cam_BodyImu_find(int64_t rigIndex, int s) {
  const auto it = rigCamToExtrIndex_.find({rigIndex, s});
  return it != rigCamToExtrIndex_.end() ? &Ts_Cam_BodyImu_[it->second] : nullptr;
}

const ExtrinsicsVariable* SingleSessionProblem::T_Cam_BodyImu_find(int64_t rigIndex, int s) const {
  const auto it = rigCamToExtrIndex_.find({rigIndex, s});
  return it != rigCamToExtrIndex_.end() ? &Ts_Cam_BodyImu_[it->second] : nullptr;
}

int64_t SingleSessionProblem::imuCalib_addNew(
    const ImuCalibParam& data,
    int64_t averageTimestampUs,
    int64_t prevVarIndex,
    int sensorIndex,
    bool constant) {
  imuCalibs_.push_back(
      ImuCalibrationVariable{
          .var = data,
          .averageTimestampUs = averageTimestampUs,
          .prevVarIndex = prevVarIndex,
          .sensorIndex = sensorIndex,
      });
  imuCalibs_.back().var.setConstant(constant);
  return imuCalibs_.size() - 1;
}

ImuCalibrationVariable& SingleSessionProblem::imuCalib_at(int index) {
  return imuCalibs_[index];
}

const ImuCalibrationVariable& SingleSessionProblem::imuCalib_at(int index) const {
  return imuCalibs_[index];
}

const SingleSessionProblem::RigSensorToIndex& SingleSessionProblem::rigImuToCalibIndex() const {
  return rigImuToCalibIndex_;
}

void SingleSessionProblem::imuCalib_setRef(int64_t rigIndex, int s, int index) {
  rigImuToCalibIndex_[{rigIndex, s}] = index;
}

ImuCalibrationVariable& SingleSessionProblem::imuCalib(int64_t rigIndex, int s) {
  return imuCalibs_[findOrDie(rigImuToCalibIndex_, {rigIndex, s})];
}

const ImuCalibrationVariable& SingleSessionProblem::imuCalib(int64_t rigIndex, int s) const {
  return imuCalibs_[findOrDie(rigImuToCalibIndex_, {rigIndex, s})];
}

ImuCalibrationVariable* SingleSessionProblem::imuCalib_find(int64_t rigIndex, int s) {
  const auto it = rigImuToCalibIndex_.find({rigIndex, s});
  return it != rigImuToCalibIndex_.end() ? &imuCalibs_[it->second] : nullptr;
}

const ImuCalibrationVariable* SingleSessionProblem::imuCalib_find(int64_t rigIndex, int s) const {
  const auto it = rigImuToCalibIndex_.find({rigIndex, s});
  return it != rigImuToCalibIndex_.end() ? &imuCalibs_[it->second] : nullptr;
}

int64_t SingleSessionProblem::T_Imu_BodyImu_addNew(
    const SE3& data,
    int64_t averageTimestampUs,
    int64_t prevVarIndex,
    int sensorIndex,
    bool constant) {
  Ts_Imu_BodyImu_.push_back(
      ExtrinsicsVariable{
          .var = data,
          .averageTimestampUs = averageTimestampUs,
          .prevVarIndex = prevVarIndex,
          .sensorIndex = sensorIndex,
      });
  Ts_Imu_BodyImu_.back().var.setConstant(constant);
  return Ts_Imu_BodyImu_.size() - 1;
}

ExtrinsicsVariable& SingleSessionProblem::T_Imu_BodyImu_at(int64_t index) {
  return Ts_Imu_BodyImu_[index];
}

const ExtrinsicsVariable& SingleSessionProblem::T_Imu_BodyImu_at(int64_t index) const {
  return Ts_Imu_BodyImu_[index];
}

const SingleSessionProblem::RigSensorToIndex& SingleSessionProblem::rigImuToExtrIndex() const {
  return rigImuToExtrIndex_;
}

void SingleSessionProblem::T_Imu_BodyImu_setRef(int64_t rigIndex, int s, int index) {
  rigImuToExtrIndex_[{rigIndex, s}] = index;
}

ExtrinsicsVariable& SingleSessionProblem::T_Imu_BodyImu(int64_t rigIndex, int s) {
  return Ts_Imu_BodyImu_[findOrDie(rigImuToExtrIndex_, {rigIndex, s})];
}

const ExtrinsicsVariable& SingleSessionProblem::T_Imu_BodyImu(int64_t rigIndex, int s) const {
  return Ts_Imu_BodyImu_[findOrDie(rigImuToExtrIndex_, {rigIndex, s})];
}

ExtrinsicsVariable* SingleSessionProblem::T_Imu_BodyImu_find(int64_t rigIndex, int s) {
  const auto it = rigImuToExtrIndex_.find({rigIndex, s});
  return it != rigImuToExtrIndex_.end() ? &Ts_Imu_BodyImu_[it->second] : nullptr;
}

const ExtrinsicsVariable* SingleSessionProblem::T_Imu_BodyImu_find(int64_t rigIndex, int s) const {
  const auto it = rigImuToExtrIndex_.find({rigIndex, s});
  return it != rigImuToExtrIndex_.end() ? &Ts_Imu_BodyImu_[it->second] : nullptr;
}

int64_t SingleSessionProblem::pointTrack_addNew(const Vec3& pt) {
  pointTracks_.emplace_back(pt);
  pointHosts_.emplace_back();
  return pointTracks_.size() - 1;
}

void SingleSessionProblem::pointTrack_setHost(
    int64_t trackIndex,
    int64_t hostRigIndex,
    bool inverseDepth) {
  XR_CHECK_LT(trackIndex, pointTracks_.size());
  PointHost& host = pointHosts_[trackIndex];
  XR_CHECK_LT(host.rigIndex, 0, "Point track {} already has a host", trackIndex);
  auto& var = pointTracks_[trackIndex];
  const Vec3 ptHost = findOrDie(inertialPoses_, hostRigIndex).T_bodyImu_world.value * var.value;
  host.rigIndex = hostRigIndex;
  host.inverseDepth = inverseDepth;
  if (inverseDepth) {
    // R0: any rotation whose third axis is the direction of the point in the host
    const double dist = ptHost.norm();
    const Vec3 d = ptHost / dist;
    const Vec3 e1 = d.cross(std::abs(d[0]) < 0.9 ? Vec3::UnitX() : Vec3::UnitY()).normalized();
    host.R0.col(0) = e1;
    host.R0.col(1) = d.cross(e1);
    host.R0.col(2) = d;
    var.value = Vec3(0.0, 0.0, 1.0 / dist);
  } else {
    host.R0.setIdentity();
    var.value = ptHost;
  }
}

const PointHost& SingleSessionProblem::pointTrack_host(int64_t trackIndex) const {
  XR_CHECK_LT(trackIndex, pointHosts_.size());
  return pointHosts_[trackIndex];
}

Vec3 SingleSessionProblem::pointTrack_world(int64_t trackIndex) const {
  XR_CHECK_LT(trackIndex, pointTracks_.size());
  const Vec3& v = pointTracks_[trackIndex].value;
  const PointHost& host = pointHosts_[trackIndex];
  if (host.rigIndex < 0) {
    return v;
  }
  const Vec3 ptHost = host.inverseDepth ? Vec3(host.R0 * Vec3(v[0], v[1], 1.0) / v[2]) : v;
  return findOrDie(inertialPoses_, host.rigIndex).T_bodyImu_world.value.inverse() * ptHost;
}

PointVariable& SingleSessionProblem::pointTrack(int64_t trackIndex) {
  XR_CHECK_LT(trackIndex, pointTracks_.size());
  return pointTracks_[trackIndex];
}

const PointVariable& SingleSessionProblem::pointTrack(int64_t trackIndex) const {
  XR_CHECK_LT(trackIndex, pointTracks_.size());
  return pointTracks_[trackIndex];
}

SoftLossType& SingleSessionProblem::reprojErrorLoss() {
  return reprojErrorLoss_;
}

const SoftLossType& SingleSessionProblem::reprojErrorLoss() const {
  return reprojErrorLoss_;
}

SoftLossType& SingleSessionProblem::imuErrorLoss() {
  return imuErrorLoss_;
}

const SoftLossType& SingleSessionProblem::imuErrorLoss() const {
  return imuErrorLoss_;
}

void SingleSessionProblem::detectorBiases_init(int numCameras) {
  detectorBiases_.assign(numCameras, Vec2::Zero());
}

Point2DVariable& SingleSessionProblem::detectorBias(int camIndex) {
  return detectorBiases_[camIndex];
}

const Point2DVariable& SingleSessionProblem::detectorBias(int camIndex) const {
  return detectorBiases_[camIndex];
}

std::unordered_map<int64_t, RollingShutterData>& SingleSessionProblem::rigToRSData() {
  return rigToRSData_;
}

// utils: return true if element is address of and element in std::vector
template <typename T, typename V>
static bool belongsToVec(const std::vector<T>& vec, V* el) {
  intptr_t start = (intptr_t)&vec[0], back = (intptr_t)(&vec.back()), elptr = (intptr_t)el;
  return (elptr >= start) && (elptr <= back);
}

bool SingleSessionProblem::isOwnPointVar(small_thing::VarBase* var) const {
  return belongsToVec(pointTracks_, var);
}

bool SingleSessionProblem::isOwnImuCalibVar(small_thing::VarBase* var) const {
  return belongsToVec(imuCalibs_, var);
}

bool SingleSessionProblem::isOwnImuExtrinsicsVar(small_thing::VarBase* var) const {
  return belongsToVec(Ts_Imu_BodyImu_, var);
}

bool SingleSessionProblem::isOwnCamExtrinsicsVar(small_thing::VarBase* var) const {
  return belongsToVec(Ts_Cam_BodyImu_, var);
}

bool SingleSessionProblem::isOwnCameraModelVar(small_thing::VarBase* var) const {
  return belongsToVec(cameraModels_, var);
}

void SingleSessionProblem::showHistogram(bool simpleStats) const {
  Histograms h(opt_);
  if (simpleStats) {
    h.showPixelErrors = false; // image-distance pixel reproj errors
    h.showRotVelPos = false; // separate histograms for rot/vel/pos
    h.separateSecondaryInertial = false; // separate main/secondary imu
    h.showAggregateCalibFactors = true; // one histogram for all rw/fprio factors
  }
  h.show();
};

void SingleSessionProblem::applyWorldTransformation(const SE3& T_newW_world) {
  for (auto& ptVar : pointTracks_) {
    ptVar.value = T_newW_world * ptVar.value;
  }

  SE3 T_world_newW = T_newW_world.inverse();
  for (auto& [_, inertialPoseVar] : inertialPoses_) {
    inertialPoseVar.T_bodyImu_world.value = inertialPoseVar.T_bodyImu_world.value * T_world_newW;
    inertialPoseVar.vel_world.value = T_newW_world.so3() * inertialPoseVar.vel_world.value;
  }

  // apply transformation on gravity only if we are the only owner
  if (gravityWorld_.use_count() == 1) {
    gravityWorld_->value.vec = T_newW_world.so3() * gravityWorld_->value.vec;
  }
}

int64_t SingleSessionProblem::averageTimestampOfRigsInRange(int64_t start, int64_t end) {
  double sumSec = 0.0;
  int nFound = 0;
  for (int64_t i = start; i < end; i++) {
    if (const auto* pRig = inertialPose_find(i)) {
      sumSec += pRig->timestampUs * 1e-6;
      nFound++;
    }
  }
  if (nFound == 0) {
    return -1;
  }
  return (int64_t)((sumSec / nFound) * 1e6);
}

} // namespace visual_inertial_ba
