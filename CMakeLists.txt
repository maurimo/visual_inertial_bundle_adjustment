#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#

cmake_minimum_required(VERSION 3.24)

# Eigen without vectorization/alignment requirements (simpler, and keeps results reproducible
# with earlier versions of this project).
add_compile_definitions(
  EIGEN_DONT_VECTORIZE
  EIGEN_DONT_ALIGN
)

project(VisualInertialBundleAdjustment CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

message("* Build type: " ${CMAKE_BUILD_TYPE})

set(VIBA_BUILD_TESTS ON CACHE BOOL "If on (default), tests are build")

# some fancy colors for cmake messages
if(NOT WIN32)
  string(ASCII 27 Esc)
  set(ColourReset "${Esc}[m")
  set(Cyan "${Esc}[36m")
endif()

include(FetchContent)
cmake_policy(SET CMP0077 NEW)
cmake_policy(SET CMP0135 NEW)
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.30")
  cmake_policy(SET CMP0169 OLD) # do not complain about FetchContent_Populate
endif()

# prefer STATIC
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# Eigen (header only). Pinned to the version effectively used by earlier versions of this
# project (fetched by projectaria_tools, which took precedence over the one declared here).
message("${Cyan}=============================[ Eigen ]==================================${ColourReset}")
FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/19cacd3ecb9dab73c2dd7bc39d9193e06ba92bdd/eigen-19cacd3ecb9dab73c2dd7bc39d9193e06ba92bdd.tar.gz
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()
message("* Eigen Source = " ${eigen_SOURCE_DIR})
include_directories(SYSTEM ${eigen_SOURCE_DIR})

# Sophus (header only), same remark as for Eigen.
message("${Cyan}=============================[ Sophus ]=================================${ColourReset}")
FetchContent_Declare(
  sophus
  URL https://github.com/strasdat/Sophus/archive/refs/tags/1.22.10.tar.gz
)
FetchContent_GetProperties(sophus)
if(NOT sophus_POPULATED)
  FetchContent_Populate(sophus)
endif()
message("* Sophus Source = " ${sophus_SOURCE_DIR})
include_directories(SYSTEM ${sophus_SOURCE_DIR})
add_compile_definitions(SOPHUS_USE_BASIC_LOGGING=1)

# fmt (system package if available)
message("${Cyan}==============================[ fmt ]===================================${ColourReset}")
find_package(fmt QUIET)
if(NOT fmt_FOUND)
  FetchContent_Declare(fmt URL https://github.com/fmtlib/fmt/archive/refs/tags/10.2.1.tar.gz)
  set(CMAKE_POSITION_INDEPENDENT_CODE ON)
  FetchContent_MakeAvailable(fmt)
  unset(CMAKE_POSITION_INDEPENDENT_CODE)
  message("* fmt Source = " ${fmt_SOURCE_DIR})
else()
  message("* fmt found in: " ${fmt_DIR})
endif()

# CLI11 (header only)
message("${Cyan}=============================[ CLI11 ]==================================${ColourReset}")
FetchContent_Declare(cli11 URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.4.2.tar.gz)
set(CLI11_PRECOMPILED OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(CLI11_BUILD_DOCS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(cli11)

# nlohmann json (header only)
message("${Cyan}=========================[ nlohmann-json ]==============================${ColourReset}")
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(nlohmann_json)

# dispenso for multithreading
message("${Cyan}============================[ dispenso ]================================${ColourReset}")
FetchContent_Declare(
  dispenso
  URL https://github.com/facebookincubator/dispenso/archive/refs/tags/v1.4.0.zip
)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
FetchContent_MakeAvailable(dispenso)
unset(CMAKE_POSITION_INDEPENDENT_CODE)
include_directories(${dispenso_SOURCE_DIR} ${dispenso_SOURCE_DIR}/dispenso/third-party/moodycamel)
message("* Dispenso Source = " ${dispenso_SOURCE_DIR})
message("* Dispenso Library Dir = " ${dispenso_BINARY_DIR})

# gtest
message("${Cyan}=============================[ GTest ]==================================${ColourReset}")
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
message("* GTest Source = " ${gtest_SOURCE_DIR})
message("* GTest Library Dir = " ${gtest_BINARY_DIR})
message("* GMock Source = " ${gmock_SOURCE_DIR})
message("* GMock Library Dir = " ${gmock_BINARY_DIR})
include(GoogleTest)

# Sokol (header-only graphics library)
message("${Cyan}============================[ Sokol ]====================================${ColourReset}")
FetchContent_Declare(
  sokol
  URL https://github.com/floooh/sokol/archive/c66a1f0.zip
  EXCLUDE_FROM_ALL
)
FetchContent_GetProperties(sokol)
if(NOT sokol_POPULATED)
  FetchContent_Populate(sokol)

  # Patch sokol_imgui.h to support 32-bit indices dynamically
  set(SOKOL_IMGUI_HEADER "${sokol_SOURCE_DIR}/util/sokol_imgui.h")
  file(READ "${SOKOL_IMGUI_HEADER}" SOKOL_IMGUI_CONTENT)

  # Find the line with pip_desc.index_type = SG_INDEXTYPE_UINT16; and replace it
  string(REPLACE
    "pip_desc.index_type = SG_INDEXTYPE_UINT16;"
    "pip_desc.index_type = sizeof(ImDrawIdx) == 2 ? SG_INDEXTYPE_UINT16 : SG_INDEXTYPE_UINT32;"
    SOKOL_IMGUI_CONTENT "${SOKOL_IMGUI_CONTENT}")

  file(WRITE "${SOKOL_IMGUI_HEADER}" "${SOKOL_IMGUI_CONTENT}")
  message("* Patched sokol_imgui.h for 32-bit index support")
endif()
message("* Sokol Source = " ${sokol_SOURCE_DIR})
include_directories(${sokol_SOURCE_DIR})

# ImGui
message("${Cyan}============================[ ImGui ]====================================${ColourReset}")
FetchContent_Declare(
  imgui
  URL https://github.com/ocornut/imgui/archive/refs/tags/v1.92.5-docking.zip
  EXCLUDE_FROM_ALL
)
FetchContent_GetProperties(imgui)
if(NOT imgui_POPULATED)
  FetchContent_Populate(imgui)

  # Patch imconfig.h to enable 32-bit indices
  set(IMGUI_CONFIG_HEADER "${imgui_SOURCE_DIR}/imconfig.h")
  file(READ "${IMGUI_CONFIG_HEADER}" IMGUI_CONFIG_CONTENT)

  # Uncomment the ImDrawIdx line
  string(REPLACE
    "//#define ImDrawIdx unsigned int"
    "#define ImDrawIdx unsigned int"
    IMGUI_CONFIG_CONTENT "${IMGUI_CONFIG_CONTENT}")

  file(WRITE "${IMGUI_CONFIG_HEADER}" "${IMGUI_CONFIG_CONTENT}")
  message("* Patched imconfig.h for 32-bit index support")
endif()
message("* ImGui Source = " ${imgui_SOURCE_DIR})

# ImPlot
message("${Cyan}============================[ ImPlot ]===================================${ColourReset}")
FetchContent_Declare(
  implot
  URL https://github.com/epezent/implot/archive/refs/tags/v0.17.zip
  EXCLUDE_FROM_ALL
)
FetchContent_GetProperties(implot)
if(NOT implot_POPULATED)
  FetchContent_Populate(implot)
endif()
message("* ImPlot Source = " ${implot_SOURCE_DIR})


# Baspacho (Force off: CUDA/tests/examples)
message("${Cyan}============================[ BaSpaCho ]=================================${ColourReset}")
set(BASPACHO_USE_CUBLAS OFF CACHE BOOL "If on, CUDA support is enabled" FORCE)
set(BASPACHO_BUILD_TESTS OFF CACHE BOOL "If on (default), tests are build" FORCE)
set(BASPACHO_BUILD_EXAMPLES OFF CACHE BOOL "If on (default), exampels/benchmarks are build" FORCE)

add_subdirectory("deps/baspacho")
include_directories("${PROJECT_SOURCE_DIR}/deps/baspacho")

link_directories("${CMAKE_CURRENT_BINARY_DIR}/deps/baspacho/baspacho/baspacho")

if(BUILD_SHARED_LIBS)
  set(BASPACHO_LIBRARY BaSpaCho)
else()
  set(BASPACHO_LIBRARY BaSpaCho_static)
endif()

# ##########################################################################################################
# add root to include path
include_directories("${PROJECT_SOURCE_DIR}")
include_directories("${PROJECT_SOURCE_DIR}/deps")

# enable tests, unless explicitly disabled
if(VIBA_BUILD_TESTS)
  enable_testing()
endif()

add_subdirectory("lib")

add_subdirectory("interfaces")

add_subdirectory("viba")
