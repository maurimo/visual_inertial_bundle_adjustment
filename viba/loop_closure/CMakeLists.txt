# Loop closure on top of the open-loop VI-BA: candidates, 5-point verification, pose graph
# (stages 1 and 2, two variants), mapping of the corrections onto the trajectory.

add_library(ark_loop_closure
    Bearing.h
    DriftMapping.h
    FivePoint.h
    LoopClosure.h
    MatchSource.h
    OdometryNoise.h
    PoseGraph.h
    DriftMapping.cpp
    FivePoint.cpp
    LoopClosure.cpp
    MatchSource.cpp
    PoseGraph.cpp
)
set_property(TARGET ark_loop_closure PROPERTY POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ark_loop_closure ark_problem fmt::fmt)
target_include_directories(ark_loop_closure PUBLIC "../..")

if(VIBA_BUILD_TESTS)
  add_subdirectory(tests)
endif()
