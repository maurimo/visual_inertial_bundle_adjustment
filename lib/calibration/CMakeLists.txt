# Camera projection models and sensor calibration types (extracted from projectaria_tools)

add_library(calibration
  CameraCalibration.cpp
  CameraCalibration.h
  CameraProjection.cpp
  CameraProjection.h
  ImuCalibration.h
  NullRef.h
  projections/Common.h
  projections/FisheyeRadTanThinPrism.h
  projections/KannalaBrandtK3.h
  projections/Linear.h
  projections/PinholeRadTan.h
  projections/Spherical.h
  projections/UnifiedFamily.h
)
target_include_directories(calibration PUBLIC "../")
set_property(TARGET calibration PROPERTY POSITION_INDEPENDENT_CODE ON)

if(VIBA_BUILD_TESTS)
  add_subdirectory(tests)
endif()
