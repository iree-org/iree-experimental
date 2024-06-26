# Copyright 2024 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.24)

#-------------------------------------------------------------------------------
# Project configuration
#-------------------------------------------------------------------------------

project(iree-prof-tools C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(IREE_PACKAGE_ROOT_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(IREE_SAMPLES_ROOT_DIR "${IREE_PACKAGE_ROOT_DIR}/..")
message("IREE-samples dir is \"${IREE_SAMPLES_ROOT_DIR}\"")

# NOTE: The IREE source code must be cloned to a directory side-by-side with
# this project.
set(IREE_ROOT_DIR "${IREE_SAMPLES_ROOT_DIR}/../iree"
    CACHE STRING "Tell where is the local git-clone of IREE")
message("IREE dir is \"${IREE_ROOT_DIR}\"")

# Set the default build type to Release if unspecified
set(DEFAULT_CMAKE_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS
          "No build type selected, default to ${DEFAULT_CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${DEFAULT_CMAKE_BUILD_TYPE}" CACHE STRING
      "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)
endif()

if(CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
  add_compile_options(-Wno-unused-local-typedefs)
endif()

# Add the entire IREE repo as a subdirectory because iree-vis requires dialects
# which depends on most of IREE packages.
# TODO(byungchul): Reduce build time when IREE has already been built.
set(IREE_BUILD_TESTS OFF)
set(IREE_BUILD_SAMPLES OFF)
set(IREE_BUILD_BINDINGS_TFLITE OFF)
set(IREE_BUILD_BINDINGS_TFLITE_JAVA OFF)
set(IREE_HAL_DRIVER_DEFAULTS OFF)
add_subdirectory(${IREE_ROOT_DIR} iree EXCLUDE_FROM_ALL)

# Get abseil-cpp in ${CMAKE_BINARY_DIR}/third_party/abseil-cpp.
set(ABSL_PROPAGATE_CXX_STD ON)
include(FetchContent)
FetchContent_Declare(
  abseil-cpp
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG 20240116.0
  SOURCE_DIR third_party/abseil-cpp
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(abseil-cpp)
# Make command line flags available on all platforms.
add_compile_options("-DABSL_FLAGS_STRIP_NAMES=0")

option(IREE_PROF_BUILD_TRACY_DEPS "Build tracy deps." OFF)
if(IREE_PROF_BUILD_TRACY_DEPS)
  FetchContent_Declare(
    tbb
    GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
    GIT_TAG v2021.12.0-rc2
    SOURCE_DIR third_party/oneTBB
    EXCLUDE_FROM_ALL
  )
  #set(BUILD_SHARED_LIBS OFF)
  FetchContent_MakeAvailable(tbb)

  FetchContent_Declare(
    zstd
    GIT_REPOSITORY https://github.com/facebook/zstd.git
    GIT_TAG v1.5.6
    SOURCE_DIR third_party/zstd
    SOURCE_SUBDIR build/cmake
    EXCLUDE_FROM_ALL
  )
  set(ZSTD_BUILD_PROGRAMS OFF)
  FetchContent_MakeAvailable(zstd)
  add_library(zstd ALIAS libzstd_static)

  FetchContent_Declare(
    capstone
    GIT_REPOSITORY https://github.com/libcapstone/libcapstone.git
    GIT_TAG 4.0.2
    SOURCE_DIR third_party/libcapstone
    EXCLUDE_FROM_ALL
  )
  FetchContent_MakeAvailable(capstone)
  add_library(capstone ALIAS capstone-static)
endif()

# Add tracy in IREE repo since tracy should be disabled for performance.
add_subdirectory(${IREE_ROOT_DIR}/build_tools/third_party/tracy
                 third_party/tracy EXCLUDE_FROM_ALL)

include_directories(
  ${IREE_SAMPLES_ROOT_DIR} ${IREE_ROOT_DIR} ${CMAKE_BINARY_DIR}
)

#-------------------------------------------------------------------------------
# Build targets
#-------------------------------------------------------------------------------

iree_cc_library(
  NAME
    iree-prof-output
  HRDS
    "iree-prof-output.h"
    "iree-prof-output-chrome.h"
    "iree-prof-output-stdout.h"
    "iree-prof-output-tracy.h"
    "iree-prof-output-utils.h"
  SRCS
    "iree-prof-output.cc"
    "iree-prof-output-chrome.cc"
    "iree-prof-output-stdout.cc"
    "iree-prof-output-tracy.cc"
    "iree-prof-output-utils.cc"
  DEPS
    IREETracyServer
    absl::check
    absl::flags
    absl::flags_parse
    absl::flat_hash_map
    absl::log
    absl::log_initialize
    absl::log_severity
    absl::status
    absl::strings
    absl::time
)

iree_cc_binary(
  NAME
    iree-prof
  SRCS
    "iree-prof.cc"
  DEPS
    ::iree-prof-output
    IREETracyServer
    absl::flags
    absl::log
    absl::strings
    $<LINK_LIBRARY:WHOLE_ARCHIVE,absl::log_flags>
)


iree_cc_binary(
  NAME
    iree-prof-convert
  SRCS
    "iree-prof-convert.cc"
  DEPS
    ::iree-prof-output
    IREETracyServer
    absl::flags
    absl::log
    $<LINK_LIBRARY:WHOLE_ARCHIVE,absl::log_flags>
)

if(IREE_BUILD_COMPILER)
  iree_cc_library(
    NAME
      iree-vis-graph
    SRCS
      "graph.cc"
      "graph-util.cc"
    HDRS
      "graph.h"
      "graph-util.h"
    DEPS
      LLVMSupport
      MLIRParser
      absl::check
      absl::flat_hash_map
      absl::log
      absl::status
      absl::statusor
      absl::strings
      iree::compiler::Dialect::Stream::IR
      iree::compiler::Dialect::Util::IR
  )
    
  iree_cc_binary(
    NAME
      iree-vis
    SRCS
      "iree-vis.cc"
    DEPS
      ::iree-vis-graph
      absl::flags
      absl::flags_parse
      absl::log
      absl::log_initialize
      absl::log_severity
      iree::compiler::API::StaticImpl
      iree::compiler::bindings::c::headers
      $<LINK_LIBRARY:WHOLE_ARCHIVE,absl::log_flags>
  )
endif()
