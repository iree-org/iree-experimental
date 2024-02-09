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
message("IREE dir is ${IREE_ROOT_DIR}")

# Load useful IREE CMake modules.
list(APPEND CMAKE_MODULE_PATH ${IREE_ROOT_DIR}/build_tools/cmake/)
include(iree_macros)
include(iree_cc_binary)
include(iree_cc_library)
include(iree_install_support)
include(external_cc_library)

# Add third_party directories and libraries.
set(ABSL_PROPAGATE_CXX_STD ON)
add_subdirectory(${IREE_SAMPLES_ROOT_DIR}/third_party/abseil-cpp
                 third_party/abseil-cpp EXCLUDE_FROM_ALL)

# set(TRACY_WORKER_SRCS
#     ${IREE_ROOT_DIR}/third_party/tracy/server/TracyWorker.cpp
# )
# add_library(tracy_worker ${TRACY_WORKER_SRCS})
# add_library(tracy::worker ALIAS tracy_worker)
add_subdirectory(${IREE_ROOT_DIR}/build_tools/third_party/tracy
                 third_party/tracy EXCLUDE_FROM_ALL)

include_directories(${IREE_SAMPLES_ROOT_DIR} ${IREE_ROOT_DIR})

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
    absl::flags
    absl::flags_parse
    absl::flat_hash_map
    absl::log
    absl::log_initialize
    absl::log_severity
    absl::status
    absl::strings
    absl::time
    tracy::worker
)

iree_cc_binary(
  NAME
    iree-prof
  SRCS
    "iree-prof.cc"
  DEPS
    ::iree-prof-output
    absl::flags
    absl::log
    absl::strings
    tracy::worker
    $<LINK_LIBRARY:WHOLE_ARCHIVE,absl::log_flags>
)

iree_cc_binary(
  NAME
    iree-prof-convert
  SRCS
    "iree-prof-convert.cc"
  DEPS
    ::iree-prof-output
    absl::flags
    absl::log
    tracy::worker
    $<LINK_LIBRARY:WHOLE_ARCHIVE,absl::log_flags>
)
