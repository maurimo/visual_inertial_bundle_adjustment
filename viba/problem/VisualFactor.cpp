/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <viba/problem/SingleSessionProblem.h>

#include <viba/problem/VisualFactorRelative.h>

#include <rolling_shutters/RollingShutterData.h>
#include <viba/common/Enumerate.h>

#include <logging/Checks.h>
#define DEFAULT_LOG_CHANNEL "ViBa::VisualFactor"
#include <logging/Log.h>

namespace visual_inertial_ba {

void SingleSessionProblem::printDetectorBiases(const std::string& label) {
  printDetectorBiases(label, detectorBiases_);
}

void SingleSessionProblem::printDetectorBiases(
    const std::string& label,
    const std::vector<Point2DVariable>& vars) {
  std::stringstream ss;
  ss << "Detector bias (" << label << "):" << std::endl;
  for (const auto& [i, var] : enumerate(vars)) {
    ss << "  Camera n." << i << ": BX=" << var.value[0] << ", BY=" << var.value[1] << std::endl;
  }
  XR_LOGI("{}", ss.str());
}

// Note: don't put in anonymous namespace so we can find the factors via introspection
// @lint-ignore-every CLANGTIDY facebook-hte-ShadowingClass
struct VisualFactor {
  VisualFactor(const Vec2& projBaseRes, const Mat22& sqrtH_BaseRes)
      : projBaseRes_(projBaseRes), sqrtH_BaseRes_(sqrtH_BaseRes) {}

  std::optional<Vec2> operator()(
      const Vec3& worldPt,
      const SE3& T_bodyImu_world,
      const SE3& T_Cam_BodyImu,
      const CameraModelParam& cameraModel,
      Ref<Mat23> worldPt_Jacobian,
      Ref<Mat26> T_bodyImu_world_Jacobian,
      Ref<Mat26> T_Cam_BodyImu_Jacobian,
      Ref<Mat<2, Eigen::Dynamic>> cameraModel_Jacobian) const {
    const Vec3 pointKeyRig = T_bodyImu_world * worldPt;
    const Vec3 pointCamera = T_Cam_BodyImu * pointKeyRig;

    Vec2 proj;
    Mat23 dProj_dPointCam;
    const int nIntrinsics = cameraModel.numModelParameters();
    Mat<2, Eigen::Dynamic> dProj_dCamParams(2, nIntrinsics); // TODO: avoid dynamic allocation
    if (!cameraModel.project(pointCamera, proj, dProj_dPointCam, dProj_dCamParams)) {
      return {};
    }

    const Vec2 error = proj - projBaseRes_;
    const Vec2 whiteErr = sqrtH_BaseRes_ * error;
    const Mat23 dWhiteErr_dPointCam = sqrtH_BaseRes_ * dProj_dPointCam;
    // clang-format off
    if (!isNull(worldPt_Jacobian)) {
      worldPt_Jacobian = //
        dWhiteErr_dPointCam * (T_Cam_BodyImu.so3() * T_bodyImu_world.so3()).matrix();
    }
    if (!isNull(T_bodyImu_world_Jacobian)) {
      const Mat23 dWhiteErr_dRigPtT = dWhiteErr_dPointCam * T_Cam_BodyImu.so3().matrix();
      T_bodyImu_world_Jacobian << //
        dWhiteErr_dRigPtT, dWhiteErr_dRigPtT * SO3::hat(-pointKeyRig);
    }
    if (!isNull(T_Cam_BodyImu_Jacobian)) {
      T_Cam_BodyImu_Jacobian << //
        dWhiteErr_dPointCam, dWhiteErr_dPointCam * SO3::hat(-pointCamera);
    }
    if (!isNull(cameraModel_Jacobian)) {
      cameraModel_Jacobian.leftCols(nIntrinsics) = sqrtH_BaseRes_ * dProj_dCamParams;
    }
    // clang-format on
    return whiteErr;
  }

  std::optional<Vec2> operator()(
      const Vec3& worldPt,
      const SE3& T_bodyImu_world,
      const SE3& T_Cam_BodyImu,
      const CameraModelParam& cameraModel,
      const Vec2& detectorBias,
      Ref<Mat23> worldPt_Jacobian,
      Ref<Mat26> T_bodyImu_world_Jacobian,
      Ref<Mat26> T_Cam_BodyImu_Jacobian,
      Ref<Mat<2, Eigen::Dynamic>> cameraModel_Jacobian,
      Ref<Mat22> detectorBias_Jacobian) const {
    std::optional maybeErr = this->operator()(
        worldPt,
        T_bodyImu_world,
        T_Cam_BodyImu,
        cameraModel,
        std::move(worldPt_Jacobian),
        std::move(T_bodyImu_world_Jacobian),
        std::move(T_Cam_BodyImu_Jacobian),
        std::move(cameraModel_Jacobian));
    if (!maybeErr.has_value()) {
      return {};
    }
    if (!isNull(detectorBias_Jacobian)) {
      detectorBias_Jacobian = sqrtH_BaseRes_;
    }
    return sqrtH_BaseRes_ * detectorBias + *maybeErr;
  }

  double imageRow() const {
    return projBaseRes_[1];
  }

 private:
  Vec2 projBaseRes_;
  Mat22 sqrtH_BaseRes_;
};

struct RollingShutterVisualFactor {
  static constexpr bool kUseAnalyticTimeOffsetJac = false;

  RollingShutterVisualFactor(
      const Vec2& projBaseRes,
      const Mat22& sqrtH_BaseRes,
      const RollingShutterData& rsData,
      const double* extraDtSec = nullptr)
      : visFactor_(projBaseRes, sqrtH_BaseRes), rsData_(rsData), extraDtSec_(extraDtSec) {}

  std::optional<Vec2> operator()(
      const Vec3& worldPt,
      const SE3& T_bodyImu_world,
      const SE3& T_Cam_BodyImu,
      const CameraModelParam& cameraModel,
      const Vec3& vel_world,
      Ref<Mat23> worldPt_Jacobian,
      Ref<Mat26> T_bodyImu_world_Jacobian,
      Ref<Mat26> T_Cam_BodyImu_Jacobian,
      Ref<Mat<2, Eigen::Dynamic>> cameraModel_Jacobian,
      Ref<Mat23> vel_world_Jacobian) const {
    const double timeParamFactor = visFactor_.imageRow() / cameraModel.imageHeight() - 0.5;
    const double dtSec = cameraModel.readoutTimeSec() * timeParamFactor -
        cameraModel.timeOffsetSec_Dev_Camera() + (extraDtSec_ != nullptr ? *extraDtSec_ : 0.0);

    const bool needVelocities = kUseAnalyticTimeOffsetJac && !isNull(cameraModel_Jacobian) &&
        (cameraModel.estimateReadoutTime || cameraModel.estimateTimeOffset);

    const RollingShutterEstimate shiftedEst =
        rsData_.getEstimate(dtSec, vel_world, T_bodyImu_world.inverse(), needVelocities);
    const SE3 T_bodyImuAtT_bodyImuMid = shiftedEst.T_midImu_imuAtT.inverse();
    const SE3 T_bodyImuAtT_world = T_bodyImuAtT_bodyImuMid * T_bodyImu_world;

    Mat26 T_bodyImuAtT_world_Jacobian;
    const bool dontNeedPoseJacobian =
        (isNull(cameraModel_Jacobian) ||
         !(cameraModel.estimateReadoutTime || cameraModel.estimateTimeOffset)) &&
        isNull(T_bodyImu_world_Jacobian) && isNull(vel_world_Jacobian);
    const auto maybeErr = visFactor_(
        worldPt,
        T_bodyImuAtT_world,
        T_Cam_BodyImu,
        cameraModel,
        worldPt_Jacobian,
        dontNeedPoseJacobian ? NullRef() : Ref<Mat26>(T_bodyImuAtT_world_Jacobian),
        T_Cam_BodyImu_Jacobian,
        cameraModel_Jacobian);

    if (!isNull(cameraModel_Jacobian) &&
        (cameraModel.estimateReadoutTime || cameraModel.estimateTimeOffset)) {
      Vec6 dT_bodyImuAtT_world_dT;
      if constexpr (kUseAnalyticTimeOffsetJac) {
        dT_bodyImuAtT_world_dT << -shiftedEst.vel_imuT_atT, -shiftedEst.omega_imuT_atT;
      } else {
        constexpr double kEpsilon = 1e-6;
        const RollingShutterEstimate shiftedEstP =
            rsData_.getEstimate(dtSec + kEpsilon, vel_world, T_bodyImu_world.inverse(), false);
        const SE3 T_bodyImuAtTpEps_bodyImuAtT =
            shiftedEstP.T_midImu_imuAtT.inverse() * shiftedEst.T_midImu_imuAtT;
        dT_bodyImuAtT_world_dT = T_bodyImuAtTpEps_bodyImuAtT.log() / kEpsilon; /* NUMERIC */
      }

      const Vec2 dErr_dT = T_bodyImuAtT_world_Jacobian * dT_bodyImuAtT_world_dT;
      int idx = cameraModel_Jacobian.cols();
      if (cameraModel.estimateTimeOffset) {
        cameraModel_Jacobian.col(--idx) = -dErr_dT;
      }
      if (cameraModel.estimateReadoutTime) {
        cameraModel_Jacobian.col(--idx) = dErr_dT * timeParamFactor;
      }
    }

    if (!isNull(T_bodyImu_world_Jacobian)) {
      Mat66 dT_bodyImuAtT_world_dBodyImuMid = T_bodyImuAtT_bodyImuMid.Adj();
      dT_bodyImuAtT_world_dBodyImuMid.topRightCorner<3, 3>() += SO3::hat(
          T_bodyImuAtT_world.so3() *
          (vel_world * dtSec + 0.5 * dtSec * dtSec * rsData_.gravityWorld()));

      T_bodyImu_world_Jacobian = T_bodyImuAtT_world_Jacobian * dT_bodyImuAtT_world_dBodyImuMid;
    }

    if (!isNull(vel_world_Jacobian)) {
      Mat63 dT_bodyImuAtT_world_dVelW;
      dT_bodyImuAtT_world_dVelW << T_bodyImuAtT_world.so3().matrix() * (-dtSec), Mat33::Zero();

      vel_world_Jacobian = T_bodyImuAtT_world_Jacobian * dT_bodyImuAtT_world_dVelW;
    }

    return maybeErr;
  }

  const VisualFactor visFactor_;
  const RollingShutterData& rsData_;
  // a time added to that of this observation (nullptr: none), see addObservationTimeShift
  const double* extraDtSec_;
};

// The pose of the body at the time an observation was taken (its image row and the time offset of
// the camera) from the pose at the time of the frame, with what the rolling-shutter factors for
// points expressed in a host rig need of its derivatives. Same model as RollingShutterVisualFactor.
struct RollingShutterShift {
  RollingShutterShift(
      const RollingShutterData& rsData,
      double imageRow,
      const CameraModelParam& cameraModel,
      const Vec3& vel_world,
      const SE3& T_bodyImu_world,
      const double* extraDtSec)
      : timeParamFactor(imageRow / cameraModel.imageHeight() - 0.5),
        dtSec(
            cameraModel.readoutTimeSec() * timeParamFactor -
            cameraModel.timeOffsetSec_Dev_Camera() +
            (extraDtSec != nullptr ? *extraDtSec : 0.0)) {
    const RollingShutterEstimate shiftedEst =
        rsData.getEstimate(dtSec, vel_world, T_bodyImu_world.inverse(), false);
    T_bodyImuAtT_bodyImuMid = shiftedEst.T_midImu_imuAtT.inverse();
    T_bodyImuAtT_world = T_bodyImuAtT_bodyImuMid * T_bodyImu_world;
    // The part of the displacement that turns with the body, u = R_mid_world * (v dt + g dt^2 / 2):
    // T_bodyImuMid_bodyImuAtT has the translation (inertial part) + u, so that with the pose at
    // the frame's time perturbed on the left by (dt, dw) the shift T_bodyImuAtT_bodyImuMid is
    // perturbed on the left by (R_shift * hat(u) * dw, 0).
    // (RollingShutterVisualFactor uses hat(R_shift * u) instead, the same to first order in the
    // rotation over dt; here it is the only term of the host's own observations)
    dShift_dRotMid = T_bodyImuAtT_bodyImuMid.so3().matrix() *
        SO3::hat(
            T_bodyImu_world.so3() *
            (vel_world * dtSec + 0.5 * dtSec * dtSec * rsData.gravityWorld()));

    constexpr double kEpsilon = 1e-6;
    const RollingShutterEstimate shiftedEstP =
        rsData.getEstimate(dtSec + kEpsilon, vel_world, T_bodyImu_world.inverse(), false);
    dPose_dTime = (shiftedEstP.T_midImu_imuAtT.inverse() * shiftedEst.T_midImu_imuAtT).log() /
        kEpsilon; /* NUMERIC */
  }

  static bool hasTimeColumns(const CameraModelParam& cameraModel) {
    return cameraModel.estimateReadoutTime || cameraModel.estimateTimeOffset;
  }

  // poseAtT_Jacobian: d err / d (left perturbation of the pose at the time of the observation)
  void timeColumns(
      const CameraModelParam& cameraModel,
      const Mat26& poseAtT_Jacobian,
      Ref<Mat<2, Eigen::Dynamic>> cameraModel_Jacobian) const {
    const Vec2 dErr_dT = poseAtT_Jacobian * dPose_dTime;
    int idx = cameraModel_Jacobian.cols();
    if (cameraModel.estimateTimeOffset) {
      cameraModel_Jacobian.col(--idx) = -dErr_dT;
    }
    if (cameraModel.estimateReadoutTime) {
      cameraModel_Jacobian.col(--idx) = dErr_dT * timeParamFactor;
    }
  }

  Mat23 velocityJacobian(const Mat26& poseAtT_Jacobian) const {
    return poseAtT_Jacobian.leftCols<3>() * T_bodyImuAtT_world.so3().matrix() * (-dtSec);
  }

  double timeParamFactor;
  double dtSec;
  SE3 T_bodyImuAtT_bodyImuMid;
  SE3 T_bodyImuAtT_world;
  Mat33 dShift_dRotMid;
  Vec6 dPose_dTime;
};

// Rolling shutter / camera time offset, point expressed in a host rig (VisualFactorRelative.h),
// observation from another rig. The point is in the body frame of the host at the time of the
// host's frame (the pose variable of the host), whatever the rows of the host's own observations;
// the observing body is taken at the time of the observation.
struct RollingShutterVisualFactorHosted {
  RollingShutterVisualFactorHosted(
      const Vec2& projBaseRes,
      const Mat22& sqrtH_BaseRes,
      bool inverseDepth,
      const Mat33& R0,
      const RollingShutterData& rsData,
      const double* extraDtSec = nullptr)
      : visFactor_(projBaseRes, sqrtH_BaseRes, inverseDepth, R0),
        rsData_(rsData),
        extraDtSec_(extraDtSec) {}

  std::optional<Vec2> operator()(
      const Vec3& pt,
      const SE3& T_hostBodyImu_world,
      const SE3& T_bodyImu_world,
      const SE3& T_Cam_BodyImu,
      const CameraModelParam& cameraModel,
      const Vec3& vel_world,
      Ref<Mat23> pt_Jacobian,
      Ref<Mat26> T_hostBodyImu_world_Jacobian,
      Ref<Mat26> T_bodyImu_world_Jacobian,
      Ref<Mat26> T_Cam_BodyImu_Jacobian,
      Ref<Mat<2, Eigen::Dynamic>> cameraModel_Jacobian,
      Ref<Mat23> vel_world_Jacobian) const {
    const RollingShutterShift shift(
        rsData_, visFactor_.imageRow(), cameraModel, vel_world, T_bodyImu_world, extraDtSec_);
    const bool timeColumns =
        !isNull(cameraModel_Jacobian) && RollingShutterShift::hasTimeColumns(cameraModel);
    const bool needPoseJacobian =
        timeColumns || !isNull(T_bodyImu_world_Jacobian) || !isNull(vel_world_Jacobian);
    Mat26 poseAtT_Jacobian;
    const auto maybeErr = visFactor_(
        pt,
        T_hostBodyImu_world,
        shift.T_bodyImuAtT_world,
        T_Cam_BodyImu,
        cameraModel,
        pt_Jacobian,
        T_hostBodyImu_world_Jacobian,
        needPoseJacobian ? Ref<Mat26>(poseAtT_Jacobian) : NullRef(),
        T_Cam_BodyImu_Jacobian,
        cameraModel_Jacobian);
    if (!maybeErr.has_value()) {
      return {};
    }
    if (timeColumns) {
      shift.timeColumns(cameraModel, poseAtT_Jacobian, cameraModel_Jacobian);
    }
    if (!isNull(T_bodyImu_world_Jacobian)) {
      Mat66 dPoseAtT_dPoseMid = shift.T_bodyImuAtT_bodyImuMid.Adj();
      dPoseAtT_dPoseMid.topRightCorner<3, 3>() += shift.dShift_dRotMid;
      T_bodyImu_world_Jacobian = poseAtT_Jacobian * dPoseAtT_dPoseMid;
    }
    if (!isNull(vel_world_Jacobian)) {
      vel_world_Jacobian = shift.velocityJacobian(poseAtT_Jacobian);
    }
    return maybeErr;
  }

  const VisualFactorHosted visFactor_;
  const RollingShutterData& rsData_;
  const double* extraDtSec_;
};

// Rolling shutter / camera time offset, point expressed in a host rig, observation from the host
// rig itself: the point (in the body at the time of the frame) is seen from the body at the time
// of the observation, T_bodyImuAtT_bodyImuMid away. That motion comes from the inertial samples
// around the frame, the velocity and gravity; of the pose only the attitude enters (it turns
// velocity and gravity into the body).
struct RollingShutterVisualFactorHostRig {
  RollingShutterVisualFactorHostRig(
      const Vec2& projBaseRes,
      const Mat22& sqrtH_BaseRes,
      bool inverseDepth,
      const Mat33& R0,
      const RollingShutterData& rsData,
      const double* extraDtSec = nullptr)
      : visFactor_(projBaseRes, sqrtH_BaseRes, inverseDepth, R0),
        rsData_(rsData),
        extraDtSec_(extraDtSec) {}

  std::optional<Vec2> operator()(
      const Vec3& pt,
      const SE3& T_bodyImu_world,
      const SE3& T_Cam_BodyImu,
      const CameraModelParam& cameraModel,
      const Vec3& vel_world,
      Ref<Mat23> pt_Jacobian,
      Ref<Mat26> T_bodyImu_world_Jacobian,
      Ref<Mat26> T_Cam_BodyImu_Jacobian,
      Ref<Mat<2, Eigen::Dynamic>> cameraModel_Jacobian,
      Ref<Mat23> vel_world_Jacobian) const {
    const RollingShutterShift shift(
        rsData_, visFactor_.imageRow(), cameraModel, vel_world, T_bodyImu_world, extraDtSec_);
    const bool timeColumns =
        !isNull(cameraModel_Jacobian) && RollingShutterShift::hasTimeColumns(cameraModel);
    const bool needShiftJacobian =
        timeColumns || !isNull(T_bodyImu_world_Jacobian) || !isNull(vel_world_Jacobian);
    Mat26 shift_Jacobian;
    // "hosted" by a body at the identity, observed from the body T_bodyImuAtT_bodyImuMid
    const auto maybeErr = visFactor_(
        pt,
        SE3(),
        shift.T_bodyImuAtT_bodyImuMid,
        T_Cam_BodyImu,
        cameraModel,
        pt_Jacobian,
        NullRef(),
        needShiftJacobian ? Ref<Mat26>(shift_Jacobian) : NullRef(),
        T_Cam_BodyImu_Jacobian,
        cameraModel_Jacobian);
    if (!maybeErr.has_value()) {
      return {};
    }
    if (timeColumns) {
      shift.timeColumns(cameraModel, shift_Jacobian, cameraModel_Jacobian);
    }
    if (!isNull(T_bodyImu_world_Jacobian)) {
      T_bodyImu_world_Jacobian << Mat23::Zero(),
          shift_Jacobian.leftCols<3>() * shift.dShift_dRotMid;
    }
    if (!isNull(vel_world_Jacobian)) {
      vel_world_Jacobian = shift.velocityJacobian(shift_Jacobian);
    }
    return maybeErr;
  }

  const VisualFactorHosted visFactor_;
  const RollingShutterData& rsData_;
  const double* extraDtSec_;
};

void SingleSessionProblem::addVisualFactor(
    int64_t rigIndex,
    int64_t cameraIndex,
    PointVariable& pointTrackVar,
    const Vec2& projBaseRes,
    const Mat22& sqrtH_BaseRes,
    const SoftLossType& reprojErrorLoss,
    const double* timeShiftSec) {
  const int64_t cameraModelIndex = findOrDie(rigCamToModelIndex_, {rigIndex, cameraIndex});
  const int64_t cameraExtrinsicsIndex = findOrDie(rigCamToExtrIndex_, {rigIndex, cameraIndex});

  // get rig and extrinsic variables
  XR_CHECK_LT(cameraExtrinsicsIndex, Ts_Cam_BodyImu_.size());
  auto& rigVar = findOrDie(inertialPoses_, rigIndex);
  auto& T_Cam_BodyImu_var = Ts_Cam_BodyImu_[cameraExtrinsicsIndex].var;
  auto& camVar = cameraModels_[cameraModelIndex].var;

  small_thing::FactorStoreBase* factorStorePtr;
  int64_t factorIndex;

  if (camVar.value.hasTimeOffset() || camVar.value.isRollingShutter() ||
      timeShiftSec != nullptr) {
    const auto& rsData = findOrDie(rigToRSData_, rigIndex);
    std::tie(factorStorePtr, factorIndex) = opt_.addFactor(
        RollingShutterVisualFactor(projBaseRes, sqrtH_BaseRes, rsData, timeShiftSec),
        reprojErrorLoss, // soft loss (Huber/Cauchy/etc...)
        pointTrackVar,
        rigVar.T_bodyImu_world,
        T_Cam_BodyImu_var,
        camVar,
        rigVar.vel_world);
  } else {
    std::tie(factorStorePtr, factorIndex) = opt_.addFactor(
        VisualFactor(projBaseRes, sqrtH_BaseRes),
        reprojErrorLoss, // soft loss (Huber/Cauchy/etc...)
        pointTrackVar,
        rigVar.T_bodyImu_world,
        T_Cam_BodyImu_var,
        camVar);
  }

  if (verbosity_ == Verbose) {
    Vec2 err;
    bool hasErr = factorStorePtr->unweightedError(factorIndex, err);
    if (hasErr) {
      XR_LOGI("RIG/CAM: [{}, {}], RES: {}", rigIndex, cameraIndex, err);
    }
  }
}

void SingleSessionProblem::addVisualFactor(
    int64_t rigIndex,
    int64_t cameraIndex,
    int64_t pointTrackIndex,
    const Vec2& projBaseRes,
    const Mat22& sqrtH_BaseRes,
    const SoftLossType& reprojErrorLoss,
    const double* timeShiftSec) {
  auto& pointTrackVar = pointTrack(pointTrackIndex);
  const PointHost& host = pointTrack_host(pointTrackIndex);
  if (host.rigIndex < 0) { // world coordinates
    addVisualFactor(
        rigIndex,
        cameraIndex,
        pointTrackVar,
        projBaseRes,
        sqrtH_BaseRes,
        reprojErrorLoss,
        timeShiftSec);
    return;
  }

  const int64_t cameraModelIndex = findOrDie(rigCamToModelIndex_, {rigIndex, cameraIndex});
  const int64_t cameraExtrinsicsIndex = findOrDie(rigCamToExtrIndex_, {rigIndex, cameraIndex});
  XR_CHECK_LT(cameraExtrinsicsIndex, Ts_Cam_BodyImu_.size());
  auto& rigVar = findOrDie(inertialPoses_, rigIndex);
  auto& T_Cam_BodyImu_var = Ts_Cam_BodyImu_[cameraExtrinsicsIndex].var;
  auto& camVar = cameraModels_[cameraModelIndex].var;
  const bool rollingShutter = camVar.value.hasTimeOffset() || camVar.value.isRollingShutter() ||
      timeShiftSec != nullptr;

  if (rigIndex == host.rigIndex) {
    if (rollingShutter) {
      opt_.addFactor(
          RollingShutterVisualFactorHostRig(
              projBaseRes,
              sqrtH_BaseRes,
              host.inverseDepth,
              host.R0,
              findOrDie(rigToRSData_, rigIndex),
              timeShiftSec),
          reprojErrorLoss, // soft loss (Huber/Cauchy/etc...)
          pointTrackVar,
          rigVar.T_bodyImu_world,
          T_Cam_BodyImu_var,
          camVar,
          rigVar.vel_world);
    } else {
      opt_.addFactor(
          VisualFactorHostRig(projBaseRes, sqrtH_BaseRes, host.inverseDepth, host.R0),
          reprojErrorLoss, // soft loss (Huber/Cauchy/etc...)
          pointTrackVar,
          T_Cam_BodyImu_var,
          camVar);
    }
  } else {
    auto& hostVar = findOrDie(inertialPoses_, host.rigIndex);
    if (rollingShutter) {
      opt_.addFactor(
          RollingShutterVisualFactorHosted(
              projBaseRes,
              sqrtH_BaseRes,
              host.inverseDepth,
              host.R0,
              findOrDie(rigToRSData_, rigIndex),
              timeShiftSec),
          reprojErrorLoss, // soft loss (Huber/Cauchy/etc...)
          pointTrackVar,
          hostVar.T_bodyImu_world,
          rigVar.T_bodyImu_world,
          T_Cam_BodyImu_var,
          camVar,
          rigVar.vel_world);
    } else {
      opt_.addFactor(
          VisualFactorHosted(projBaseRes, sqrtH_BaseRes, host.inverseDepth, host.R0),
          reprojErrorLoss, // soft loss (Huber/Cauchy/etc...)
          pointTrackVar,
          hostVar.T_bodyImu_world,
          rigVar.T_bodyImu_world,
          T_Cam_BodyImu_var,
          camVar);
    }
  }
}

void SingleSessionProblem::addVisualFactorWithBias(
    int64_t rigIndex,
    int64_t cameraIndex,
    PointVariable& pointTrackVar,
    Point2DVariable& detectorBias,
    const Vec2& projBaseRes,
    const Mat22& sqrtH_BaseRes,
    const SoftLossType& reprojErrorLoss) {
  const int64_t cameraModelIndex = findOrDie(rigCamToModelIndex_, {rigIndex, cameraIndex});
  const int64_t cameraExtrinsicsIndex = findOrDie(rigCamToExtrIndex_, {rigIndex, cameraIndex});

  // get rig and extrinsic variables
  XR_CHECK_LT(cameraExtrinsicsIndex, Ts_Cam_BodyImu_.size());
  auto& rigVar = findOrDie(inertialPoses_, rigIndex);
  auto& T_Cam_BodyImu_var = Ts_Cam_BodyImu_[cameraExtrinsicsIndex].var;
  auto& camVar = cameraModels_[cameraModelIndex].var;

  XR_CHECK(
      !camVar.value.isRollingShutter(), "Image Bias not supported for rolling shutter cameras");

  const auto [factorStorePtr, factorIndex] = opt_.addFactor(
      VisualFactor(projBaseRes, sqrtH_BaseRes),
      reprojErrorLoss, // soft loss (Huber/Cauchy/etc...)
      pointTrackVar,
      rigVar.T_bodyImu_world,
      T_Cam_BodyImu_var,
      camVar,
      detectorBias);

  if (verbosity_ == Verbose) {
    auto err = factorStorePtr->rawError(factorIndex);
    if (err.has_value()) {
      XR_LOGI("RIG/CAM: [{}, {}], RES: {}", rigIndex, cameraIndex, err.value());
    }
  }
}

SE3 SingleSessionProblem::T_bodyImu_world_atImageRow(
    int64_t rigIndex,
    int cameraIndex,
    float imageRow) const {
  const auto& rigVar = inertialPose(rigIndex);
  const auto& camVar = cameraModel(rigIndex, cameraIndex).var;
  if (camVar.value.isRollingShutter() || camVar.value.hasTimeOffset()) {
    const auto& cameraModel = camVar.value;
    const double timeParamFactor = imageRow / cameraModel.imageHeight() - 0.5;
    const double dtSec =
        cameraModel.readoutTimeSec() * timeParamFactor - cameraModel.timeOffsetSec_Dev_Camera();

    const auto& rsData = findOrDie(rigToRSData_, rigIndex);
    const RollingShutterEstimate shiftedEst = rsData.getEstimate(
        dtSec,
        rigVar.vel_world.value,
        rigVar.T_bodyImu_world.value.inverse(),
        /* needVelocities = */ false);
    return shiftedEst.T_midImu_imuAtT.inverse() * rigVar.T_bodyImu_world.value;
  }

  return rigVar.T_bodyImu_world.value;
}

} // namespace visual_inertial_ba
