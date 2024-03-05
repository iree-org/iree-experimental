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
message("IREE-samples dir is ${IREE_SAMPLES_ROOT_DIR}")

# NOTE: The IREE source code must be cloned to a directory side-by-side with
# this project.
set(IREE_ROOT_DIR "${IREE_SAMPLES_ROOT_DIR}/../iree"
    CACHE STRING "Tell where is the local git-clone of IREE")
set(IREE_BUILD_DIR "${IREE_ROOT_DIR}/../iree-build"
    CACHE STRING "Tell where is the binary build directory of IREE")
message("IREE dir is ${IREE_ROOT_DIR}")
message("IREE build dir is ${IREE_BUILD_DIR}")

# Load useful IREE CMake modules.
list(APPEND CMAKE_MODULE_PATH ${IREE_ROOT_DIR}/build_tools/cmake/)
include(iree_macros)
include(iree_add_all_subdirs)
include(iree_cc_binary)
include(iree_cc_library)
include(iree_install_support)
include(external_cc_library)

# Add llvm and mlir in IREE repo.
include(iree_llvm)
iree_llvm_set_bundled_cmake_options()
add_subdirectory(${IREE_ROOT_DIR}/third_party/llvm-project/llvm
                 third_party/llvm-project/llvm EXCLUDE_FROM_ALL)
add_library(IREELLVMIncludeSetup INTERFACE)

# Add compiler and tracy in IREE repo.
add_subdirectory(${IREE_ROOT_DIR}/compiler
                 compiler EXCLUDE_FROM_ALL)
add_subdirectory(${IREE_ROOT_DIR}/build_tools/third_party/tracy
                 third_party/tracy EXCLUDE_FROM_ALL)

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

include_directories(
  ${IREE_SAMPLES_ROOT_DIR} ${IREE_ROOT_DIR} ${CMAKE_BINARY_DIR}
  ${IREE_ROOT_DIR}/third_party/llvm-project/llvm/include
  ${IREE_ROOT_DIR}/third_party/llvm-project/mlir/include
  ${IREE_BUILD_DIR}/llvm-project/tools/mlir/include
)

# Set the default build type to Release if unspecified
set(DEFAULT_CMAKE_BUILD_TYPE "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS
          "No build type selected, default to ${DEFAULT_CMAKE_BUILD_TYPE}")
  set(CMAKE_BUILD_TYPE "${DEFAULT_CMAKE_BUILD_TYPE}" CACHE STRING
      "Build type (default ${DEFAULT_CMAKE_BUILD_TYPE})" FORCE)
endif()

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

iree_cc_binary(
  NAME
    iree-vis
  SRCS
    "iree-vis.cc"
  DEPS
    absl::flags
    absl::flags_parse
    absl::log
    absl::log_initialize
    absl::log_severity
    absl::strings
    iree::compiler::bindings::c::headers
    $<LINK_LIBRARY:WHOLE_ARCHIVE,absl::log_flags>
    ${IREE_BUILD_DIR}/lib/libIREECompiler.so
)
