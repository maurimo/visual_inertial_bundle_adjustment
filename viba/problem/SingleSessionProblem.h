/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <camera_model/CameraModelParam.h>
#include <imu_model/ImuCalibParam.h>
#include <preint/PreIntegration.h>
#include <rolling_shutters/RollingShutterData.h>
#include <small_thing/Optimizer.h>
#include <viba/common/Constants.h>
#include <viba/problem/LoopBundle.h>
#include <viba/problem/Types.h>
#include <viba/problem/YawDrift.h>
#include <deque>
#include <optional>

namespace visual_inertial_ba {

using small_thing::isNull;
using small_thing::NullRef;

using pair_hash = small_thing::pair_hash;

struct InertialPoseVariables {
  small_thing::Variable<SE3> T_bodyImu_world;
  small_thing::Variable<Vec3> vel_world;
  small_thing::Variable<Vec3> omega; // angular velocity (in bodyImu's reference frame)
  int64_t timestampUs = -1;
};

// used for window of rigs sharing the same calibration params
template <typename D>
struct GroupedCalibrationVariable {
  small_thing::Variable<D> var;
  int64_t averageTimestampUs = -1;
  int64_t prevVarIndex = -1;
  int sensorIndex = -1;
};

using PointVariable = small_thing::Variable<Vec3>;

// How a point-track variable is expressed, see VisualFactorRelative.h
struct PointHost {
  int64_t rigIndex = -1; // host rig; -1: the variable is the world position
  // (u, v, inverse depth) about R0's third axis, instead of a position in the host
  bool inverseDepth = false;
  Mat33 R0 = Mat33::Identity();
};
using Point2DVariable = small_thing::Variable<Vec2>;
using CameraModelVariable = GroupedCalibrationVariable<CameraModelParam>;
using ExtrinsicsVariable = GroupedCalibrationVariable<SE3>;
using ImuCalibrationVariable = GroupedCalibrationVariable<ImuCalibParam>;
using GravityData = small_thing::S2;
using GravityVariable = small_thing::Variable<GravityData>;

using SoftLossType = small_thing::HuberLossWithCutoff;

enum Verbosity { Muted, Normal, Verbose };

// class storing the optimization problem
class SingleSessionProblem {
 public:
  using RigSensorToIndex = std::unordered_map<RigSensorIndices, int64_t, pair_hash>;

  explicit SingleSessionProblem(small_thing::Optimizer& opt, Verbosity verbosity = Normal);

  // set all rigs to constant
  void setAllRigsConstant(bool constPose = true, bool constLinVel = true, bool constAngVel = true);

  // add visual factor
  void addVisualFactor(
      int64_t rigIndex,
      int64_t cameraIndex,
      PointVariable& pointTrackVar,
      const Vec2& projBaseRes,
      const Mat22& sqrtH_BaseRes,
      const SoftLossType& reprojErrorLoss,
      const double* timeShiftSec = nullptr);

  // timeShiftSec (optional, see addObservationTimeShift): the observation was taken that much
  // later than the frame (on top of the row time and time offset of the camera, if any); the
  // rolling-shutter factors are then used, which need the rig's RollingShutterData.
  // add visual factor for a point track given by index: the factor is chosen from the way the
  // track is expressed (world coordinates, or in a host rig, see pointTrack_setHost)
  void addVisualFactor(
      int64_t rigIndex,
      int64_t cameraIndex,
      int64_t pointTrackIndex,
      const Vec2& projBaseRes,
      const Mat22& sqrtH_BaseRes,
      const SoftLossType& reprojErrorLoss,
      const double* timeShiftSec = nullptr);

  // add visual factor, with image detector bias
  void addVisualFactorWithBias(
      int64_t rigIndex,
      int64_t cameraIndex,
      PointVariable& pointTrackVar,
      Point2DVariable& detectorBias,
      const Vec2& projBaseRes,
      const Mat22& sqrtH_BaseRes,
      const SoftLossType& reprojErrorLoss);

  // util, print variables describing detector biases
  static void printDetectorBiases(
      const std::string& label,
      const std::vector<Point2DVariable>& vars);

  // print detector biases for observations in this tracking problem
  void printDetectorBiases(const std::string& label);

  // adds an inertial factor
  void addInertialFactor(
      int64_t prevRigIndex,
      int64_t nextRigIndex,
      int imuIndex,
      const PreIntegrationData& preintegrationData,
      const SoftLossType& imuErrorLoss);

  // add a prior on omega (angular velocity)
  void addOmegaPriorFactor(int64_t rigIndex, int imuIndex, const Vec3& omegaRadSec_Imu);

  // random walk, imu calib
  void addImuCalibRWFactor(int64_t calibPrevIndex, int64_t calibNextIndex, const VecX& diagSqrtH);

  // random walk, camera intrinsics
  void addCamIntrRWFactor(int64_t calibPrevIndex, int64_t calibNextIndex, const VecX& diagSqrtH);

  // random walk, imu extrinsics
  void addImuExtrRWFactor(int64_t calibPrevIndex, int64_t calibNextIndex, const Vec6& diagSqrtH);

  // random walk, camera extrinsics
  void addCamExtrRWFactor(int64_t calibPrevIndex, int64_t calibNextIndex, const Vec6& diagSqrtH);

  // set 6DoF prior on pose
  void addPosePrior(int64_t rigIndex, const SE3& prior_T_bodyImu_world, const Mat66& H);

  // prior on imu calib
  void
  addImuPrior(int64_t imuCalibIndex, const ImuMeasurementModelParameters& prior, const VecX& diagH);

  // prior on camera intrinsics
  void
  addCamIntrinsicsPrior(int64_t camModelIndex, const CameraModelParam& prior, const VecX& diagH);

  // prior on imu extrinsics
  void addImuExtrinsicsPrior(int64_t camExtrinsicsIndex, const SE3& prior, const Vec6& diagH);

  // prior on camera extrinsics
  void addCamExtrinsicsPrior(int64_t camExtrinsicsIndex, const SE3& prior, const Vec6& diagH);

  // clear all pose priors, returns the number of factors removed
  int64_t clearPosePriors();

  // add a prior identical to currently estimated value
  void constrainPositionAndYaw(int64_t rigIndex);

  // show histogram of all error terms
  void showHistogram(bool simpleStats) const;

  // register points (for Schur complement trick)
  void registerPointVariables();

  // compute covariance of all inertial poses (9DoF) and calibration parameters
  void computeCovariances();

  // number of rigs
  int64_t numRigVariables() const;

  // number of camera model variables
  int64_t numCameraModelVariables() const;

  // number of extrinsics (imu and camera)
  int64_t numExtrinsicsVariables() const;

  // number of camera extrinsics
  int64_t numCameraExtrinsicsVariables() const;

  // number of imu extrinsics
  int64_t numImuExtrinsicsVariables() const;

  // number of imu calibration
  int64_t numImuCalibVariables() const;

  // number of point tracks
  int64_t numPointTrackVariables() const;

  // utils to possibly recover what tracking problem owns a cost term
  // bool isOwnRigVar(small_thing::VarBase* var); // unimplemented, not needed ATM

  bool isOwnPointVar(small_thing::VarBase* var) const;

  bool isOwnImuCalibVar(small_thing::VarBase* var) const;

  bool isOwnImuExtrinsicsVar(small_thing::VarBase* var) const;

  bool isOwnCamExtrinsicsVar(small_thing::VarBase* var) const;

  bool isOwnCameraModelVar(small_thing::VarBase* var) const;

  // apply transformation to whole world
  void applyWorldTransformation(const SE3& T_newW_world);

  // return index of first rig
  std::optional<int64_t> firstRigIndex() const;

  // return sorted rig indices
  std::vector<int64_t> sortedRigIndices() const;

  // get map {index -> variable covariance}
  const std::unordered_map<int64_t, Eigen::MatrixXd>& variableCovariances() const;

  // get reference to poses
  const std::unordered_map<int64_t, InertialPoseVariables>& inertialPoses() const;

  // check if pose exists
  bool inertialPose_exists(int64_t rigIndex);

  // set pose values, create if doesn't exist
  void inertialPose_set(
      int64_t rigIndex,
      const SE3& T_bodyImu_world,
      const Vec3& vel_world,
      const Vec3& omega,
      int64_t timestampUs);

  // get inertial pose
  InertialPoseVariables& inertialPose(int64_t rigIndex);

  // get inertial pose (overload)
  const InertialPoseVariables& inertialPose(int64_t rigIndex) const;

  // get inertial pose, return null if doesn't exist
  InertialPoseVariables* inertialPose_find(int64_t rigIndex);

  // get inertial pose, return null if doesn't exist (overload)
  const InertialPoseVariables* inertialPose_find(int64_t rigIndex) const;

  // return T_bodyImu_world WHEN a prescribed camera row is acquired
  // if not rolling shutter/time offset: = inertialPose(rigIndex).T_bodyImu_world.value
  // otherwise, time offset corresponding to image row is applied.
  SE3 T_bodyImu_world_atImageRow(int64_t rigIndex, int cameraIndex, float imageRow) const;

  // init new world gravity variable
  void gravityWorld_initNew(const GravityData& gravityWorld, bool constant = true);

  // set world gravity to refer a shared variable
  void gravityWorld_setRef(const std::shared_ptr<GravityVariable>& gravityWorldVar);

  // get world gravity
  GravityVariable& gravityWorld();

  // get world gravity (overload)
  const GravityVariable& gravityWorld() const;

  // Gravity as a function of time: numKnots >= 2 vectors equally spaced between the two times,
  // all starting at the world gravity; the inertial factors added afterwards use the linear
  // interpolation at their mid time (main IMU only). Every knot gets a prior at the world gravity
  // and a random walk to the next knot; the square-root informations are per world axis.
  void gravityKnots_init(
      int64_t startTimestampUs,
      int64_t endTimestampUs,
      int numKnots,
      const Vec3& priorDiagSqrtH,
      const Vec3& randomWalkDiagSqrtH);

  // rotation of the Earth: angular velocity of the world frame with respect to inertial space, in
  // world coordinates [rad/s]; set before adding the inertial factors
  void setEarthRateWorld(const Vec3& earthRateWorld) {
    earthRateWorld_ = earthRateWorld;
  }

  // number of gravity knots (0: one constant world gravity)
  int64_t numGravityKnots() const;

  // time and value of a gravity knot
  int64_t gravityKnotTimestampUs(int64_t knotIndex) const;
  const Vec3& gravityKnot(int64_t knotIndex) const;

  // gravity at a time: the interpolation of the knots, or the world gravity without knots
  Vec3 gravityAt(int64_t timestampUs) const;

  // Scale of the IMU's metric as a function of time (see InitSettings::scaleKnotSec): numKnots >= 2
  // factors equally spaced between the two times, starting at 1, with a prior at 1 and a random
  // walk between neighbours. Used by the inertial factors of the main IMU added afterwards.
  void scaleKnots_init(
      int64_t startTimestampUs,
      int64_t endTimestampUs,
      int numKnots,
      double priorSqrtH,
      double randomWalkSqrtH);
  int64_t numScaleKnots() const;
  // The knots enter the inertial factors only while active (the stage at the start of a pass).
  void scaleKnots_setActive(bool active) {
    scaleKnotsActive_ = active;
  }
  // interpolated value of the knots
  double scaleKnots_valueAt(int64_t timestampUs) const;
  // "Straighten": the trajectory is divided by the knots (positions of the IMU rebuilt from the
  // steps between consecutive rigs, velocities), the points follow (in a host rig: divided by the
  // knots at the host; in world coordinates: about the rig worldPointAnchorRig[track]), and the
  // knots are set back to 1. The calibration is left as it is.
  void scaleKnots_apply(const std::vector<int64_t>& worldPointAnchorRig);

  // Yaw-drift knots (see YawDrift.h): the rate [rad/s] at which the estimated trajectory turns
  // about the vertical, constant on each of the >= 2 intervals the recording is divided into;
  // they start at 0, with a prior at 0 (and a random walk between neighbours if randomWalkSqrtH
  // > 0). The intervals are at least minKnotSec long, and each one is long enough to contain
  // the end of every point track that starts in the one before it (trackSpansUs: first and last
  // observation time of every track), so that no track covers more than two intervals.
  // The positions of the rigs at this moment are the frozen copy the drift is applied along.
  // To be called once the rigs exist and before the factors: the inertial factors of the main
  // IMU and the visual factors of points seen from another rig than their host use the knots
  // while they are active.
  // shiftSqrtH > 0: the knots also hold a velocity [m/s] at which the positions drift (prior at
  // 0 with this weight), see YawDrift.h.
  void yawKnots_init(
      double minKnotSec,
      const std::vector<std::pair<int64_t, int64_t>>& trackSpansUs,
      double priorSqrtH,
      double randomWalkSqrtH,
      double shiftSqrtH);
  // lengths of the knots' intervals [s]
  std::vector<double> yawKnotIntervalsSec() const;
  int64_t numYawKnots() const;
  void yawKnots_setActive(bool active) {
    yawPath_.active = active;
  }
  // the true states (what the factors see): T_bodyImu_world and vel_world of every rig, in time
  // order; the variables themselves without knots or when they are not active
  struct TrueRigState {
    int64_t rigIndex;
    int64_t timestampUs;
    SE3 T_bodyImu_world;
    Vec3 vel_world;
    double psi; // integral of the rate from the first rig [rad]
  };
  std::vector<TrueRigState> yawKnots_trueStates() const;
  // The drift is put into the variables (they become the true states), the knots are set back
  // to 0 and the frozen copy of the positions is taken again; the points stay as they are in
  // their host rigs. The cost changes by the priors of the knots only. Returns the integral of
  // the rate at the last rig [rad].
  double yawKnots_apply();
  // the frozen copy of the positions taken again (the knots have to be 0)
  void yawKnots_refreeze();
  // sum of the knot values put into the trajectory by yawKnots_apply since the last call of
  // this function (for the log)
  std::vector<Vec<4>> yawKnots_takeApplied();
  // (test) the knots are set to +-rate [rad/s] and the variables changed so that the true states
  // stay what they are: the cost has to stay the same
  void yawKnots_gaugeTest(double rate);
  int64_t yawKnotTimestampUs(int64_t knotIndex) const;
  double yawKnot(int64_t knotIndex) const;
  Vec3 yawKnotShift(int64_t knotIndex) const; // velocity of the shift [m/s]
  // observations whose two rigs are more than one knot interval apart (plain factors)
  int64_t yawKnots_numTooLong() const {
    return yawNumTooLong_;
  }

  // Loop-closing points (LoopBundle.h, LC_IMPL.md): one point variable per loop landmark, in its
  // own container (adding to pointTracks_ after factors exist would move the point variables),
  // expressed in a host rig; observations from any rig, minutes apart, in the matrix-free store.
  // visitGroup: the visit of the observation; pairGroup: its loop pair (GNC weight).
  int64_t loopPoint_addNew(const Vec3& worldPt, int64_t hostRigIndex, bool inverseDepth);
  Vec3 loopPoint_world(int64_t loopPoint) const;
  int64_t numLoopPoints() const {
    return loopPoints_.size();
  }
  void addLoopObservation(
      int64_t rigIndex,
      int64_t cameraIndex,
      int64_t loopPoint,
      const Vec2& projBaseRes,
      const Mat22& sqrtH_BaseRes,
      int visitGroup,
      int pairGroup);
  // the virtual points (one per loop point and visit group observed from 2 rigs at least) and
  // their preconditioner-only factors; after all loop observations, before registerPointVariables.
  // Returns their number.
  int64_t loopVirtual_build();
  // the virtual points set from their loop points (before every linearization, knots folded)
  void loopVirtual_sync();
  LoopObservationStore* loopStore();

  // A time added to that of one observation (see addVisualFactor): storage that stays where it
  // is; the value may be changed between optimizer steps.
  double* addObservationTimeShift() {
    return &obsTimeShiftSec_.emplace_back(0.0);
  }
  int64_t scaleKnotTimestampUs(int64_t knotIndex) const;
  double scaleKnot(int64_t knotIndex) const;

  // add new camera model, return index
  int64_t cameraModel_addNew(
      CameraModelParam&& data,
      int64_t averageTimestampUs,
      int64_t prevVarIndex,
      int sensorIndex,
      bool constant = false);

  // get camera model from index
  CameraModelVariable& cameraModel_at(int64_t index);

  // get camera model from index
  const CameraModelVariable& cameraModel_at(int64_t index) const;

  // get map {(rig,cam) -> camera model index}
  const RigSensorToIndex& rigCamToModelIndex() const;

  // get index of camera model for (rig, cam)
  void cameraModel_setRef(int64_t rigIndex, int s, int index);

  // get camera model
  CameraModelVariable& cameraModel(int64_t rigIndex, int s);

  // get camera model (overload)
  const CameraModelVariable& cameraModel(int64_t rigIndex, int s) const;

  // get camera model, return null if doesn't exist
  CameraModelVariable* cameraModel_find(int64_t rigIndex, int s);

  // get camera model, return null if doesn't exist (overload)
  const CameraModelVariable* cameraModel_find(int64_t rigIndex, int s) const;

  // add new camera extrinsics, return index
  int64_t T_Cam_BodyImu_addNew(
      const SE3& data,
      int64_t averageTimestampUs,
      int64_t prevVarIndex,
      int sensorIndex,
      bool constant = false);

  // get camera extrinsics from index
  ExtrinsicsVariable& T_Cam_BodyImu_at(int64_t index);

  // get camera extrinsics from index (overload)
  const ExtrinsicsVariable& T_Cam_BodyImu_at(int64_t index) const;

  // get map {(rig,cam) -> camera extrinsics index}
  const RigSensorToIndex& rigCamToExtrIndex() const;

  // get index of camera extrinsics for (rig, cam)
  void T_Cam_BodyImu_setRef(int64_t rigIndex, int s, int index);

  // get camera extrinsics
  ExtrinsicsVariable& T_Cam_BodyImu(int64_t rigIndex, int s);

  // get camera extrinsics (overload)
  const ExtrinsicsVariable& T_Cam_BodyImu(int64_t rigIndex, int s) const;

  // get camera extrinsics, return null if doesn't exist
  const ExtrinsicsVariable* T_Cam_BodyImu_find(int64_t rigIndex, int s) const;

  // get camera extrinsics, return null if doesn't exist (overload)
  ExtrinsicsVariable* T_Cam_BodyImu_find(int64_t rigIndex, int s);

  // add new imu calibration, return index
  int64_t imuCalib_addNew(
      const ImuCalibParam& data,
      int64_t averageTimestampUs,
      int64_t prevVarIndex,
      int sensorIndex,
      bool constant = false);

  // get imu calibration from index
  ImuCalibrationVariable& imuCalib_at(int index);

  // get imu calibration from index (overload)
  const ImuCalibrationVariable& imuCalib_at(int index) const;

  // get map {(rig,cam) -> imu calib index}
  const RigSensorToIndex& rigImuToCalibIndex() const;

  // get index of imu calibration for (rig, cam)
  void imuCalib_setRef(int64_t rigIndex, int s, int index);

  // get imu calibration
  ImuCalibrationVariable& imuCalib(int64_t rigIndex, int s);

  // get imu calibration (overload)
  const ImuCalibrationVariable& imuCalib(int64_t rigIndex, int s) const;

  // get imu calibration, return null if doesn't exist
  ImuCalibrationVariable* imuCalib_find(int64_t rigIndex, int s);

  // get imu calibration, return null if doesn't exist (overload)
  const ImuCalibrationVariable* imuCalib_find(int64_t rigIndex, int s) const;

  // add new imu extrinsics, return index
  int64_t T_Imu_BodyImu_addNew(
      const SE3& data,
      int64_t averageTimestampUs,
      int64_t prevVarIndex,
      int sensorIndex,
      bool constant = false);

  // get imu extrinsics from index
  ExtrinsicsVariable& T_Imu_BodyImu_at(int64_t index);

  // get imu extrinsics from index (overload)
  const ExtrinsicsVariable& T_Imu_BodyImu_at(int64_t index) const;

  // get map {(rig,cam) -> imu extrinsics index}
  const RigSensorToIndex& rigImuToExtrIndex() const;

  // get index of imu extrinsics for (rig, cam)
  void T_Imu_BodyImu_setRef(int64_t rigIndex, int s, int index);

  // get imu extrinsics
  ExtrinsicsVariable& T_Imu_BodyImu(int64_t rigIndex, int s);

  // get imu extrinsics (overload)
  const ExtrinsicsVariable& T_Imu_BodyImu(int64_t rigIndex, int s) const;

  // get imu extrinsics, return null if doesn't exist
  ExtrinsicsVariable* T_Imu_BodyImu_find(int64_t rigIndex, int s);

  // get imu extrinsics, return null if doesn't exist
  const ExtrinsicsVariable* T_Imu_BodyImu_find(int64_t rigIndex, int s) const;

  // add new point track, return index
  int64_t pointTrack_addNew(const Vec3& pt);

  // get point track from index
  PointVariable& pointTrack(int64_t trackIndex);

  // get point track from index (overload)
  const PointVariable& pointTrack(int64_t trackIndex) const;

  // express a point track in the body frame of a rig, as a position or as (u, v, inverse depth).
  // The current value of the variable must be the world position; call before adding its factors.
  void pointTrack_setHost(int64_t trackIndex, int64_t hostRigIndex, bool inverseDepth);

  // how a point track is expressed
  const PointHost& pointTrack_host(int64_t trackIndex) const;

  // world position of a point track, whatever the way its variable is expressed
  Vec3 pointTrack_world(int64_t trackIndex) const;

  // get soft (robust) loss used for observations
  SoftLossType& reprojErrorLoss();

  // get soft (robust) loss used for observations (overload)
  const SoftLossType& reprojErrorLoss() const;

  // get soft (robust) loss used for observations
  SoftLossType& imuErrorLoss();

  // get soft (robust) loss used for observations (overload)
  const SoftLossType& imuErrorLoss() const;

  // initialize detector biases
  void detectorBiases_init(int numCameras);

  // get detector bias
  Point2DVariable& detectorBias(int camIndex);

  // get detector bias (overload)
  const Point2DVariable& detectorBias(int camIndex) const;

  // get rolling shutter data
  std::unordered_map<int64_t, RollingShutterData>& rigToRSData();

  int64_t averageTimestampOfRigsInRange(int64_t start, int64_t end);

 private:
  // data
  Verbosity verbosity_;

  // optimizer
  small_thing::Optimizer& opt_;

  SoftLossType reprojErrorLoss_{
      kReprojectionErrorHuberLossWidth,
      kReprojectionErrorHuberLossCutoff};
  SoftLossType imuErrorLoss_{kImuErrorHuberLossWidth, kImuErrorHuberLossCutoff};

  // covariances of optimized variables
  std::unordered_map<int64_t, Eigen::MatrixXd> variableCovariances_;

  // variable (reference to) with world gravity, constant unless computing marginal factors
  std::shared_ptr<GravityVariable> gravityWorld_;

  // gravity knots (optional), see gravityKnots_init
  std::vector<small_thing::Variable<Vec3>> gravityKnots_;
  int64_t gravityKnotsStartUs_ = 0;
  double gravityKnotsSpacingUs_ = 1.0;

  // index of the knot before a time, and where the time lies between it and the next one (0..1)
  std::pair<int64_t, double> gravityKnotAt(int64_t timestampUs) const;

  // angular velocity of the world frame with respect to inertial space [rad/s], world coordinates
  // (main IMU's inertial factors only)
  Vec3 earthRateWorld_ = Vec3::Zero();

  // scale knots (optional), see scaleKnots_init
  std::vector<small_thing::Variable<Vec<1>>> scaleKnots_;
  int64_t scaleKnotsStartUs_ = 0;
  double scaleKnotsSpacingUs_ = 1.0;
  std::pair<int64_t, double> scaleKnotAt(int64_t timestampUs) const;
  bool scaleKnotsActive_ = true;

  // yaw-drift knots (optional), see yawKnots_init
  std::vector<small_thing::Variable<Vec<4>>> yawKnots_; // rate [rad/s], velocity [m/s]
  int64_t yawKnotsStartUs_ = 0;
  YawDriftPath yawPath_; // the factors keep a pointer to it
  std::unordered_map<int64_t, int> yawRigOrder_; // rig index -> index in yawPath_
  int64_t yawNumTooLong_ = 0;
  std::vector<Vec<4>> yawKnotsApplied_;

  std::deque<double> obsTimeShiftSec_;

  // loop-closing points (see loopPoint_addNew)
  std::deque<PointVariable> loopPoints_;
  std::vector<PointHost> loopPointHosts_;
  struct LoopObsRecord {
    int64_t rigIndex, cameraIndex, timestampUs;
    Vec2 px;
    Mat22 sqrtH;
    int visitGroup, pairGroup;
  };
  std::vector<std::vector<LoopObsRecord>> loopObsRecords_; // per loop point
  std::deque<PointVariable> loopVirtualPoints_;
  struct LoopVirtual {
    int64_t loopPoint;
    PointHost host;
  };
  std::vector<LoopVirtual> loopVirtualInfo_;
  LoopObservationStore* loopStore_ = nullptr;
  // a point given in world coordinates, as the variable of a point in a host rig
  Vec3 hostParamOf(const PointHost& host, const Vec3& worldPt) const;

  // rigIndex (in recording's frame list) -> variable
  // (note that only a subset of the rigs in the recording might be loaded)
  std::unordered_map<int64_t, InertialPoseVariables> inertialPoses_;

  // map points
  std::vector<PointVariable> pointTracks_;
  std::vector<PointHost> pointHosts_; // same indices as pointTracks_

  // camera intrinsics
  std::vector<CameraModelVariable> cameraModels_; // per camera type, contains a vector
  RigSensorToIndex rigCamToModelIndex_;

  // camera extrinsics
  std::vector<ExtrinsicsVariable> Ts_Cam_BodyImu_;
  RigSensorToIndex rigCamToExtrIndex_;

  // imu calibration
  std::vector<ImuCalibrationVariable> imuCalibs_;
  RigSensorToIndex rigImuToCalibIndex_;

  // imu extrinsics
  std::vector<ExtrinsicsVariable> Ts_Imu_BodyImu_;
  RigSensorToIndex rigImuToExtrIndex_;

  // detector biases
  std::vector<Point2DVariable> detectorBiases_;

  // rolling shutter data
  std::unordered_map<int64_t, RollingShutterData> rigToRSData_;
};

} // namespace visual_inertial_ba
