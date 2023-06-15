# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.24)

project(OPENXLA_ASYNC)

include(CMakeDependentOption)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(OPENXLA_ASYNC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# TODO: Fix this once the project is slotted into place.
if(NOT IREE_ROOT_DIR)
  set(IREE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../iree")
endif()

set(IREE_CXX_STANDARD ${CMAKE_CXX_STANDARD})

# Enable testing before any subdirectories are added.
enable_testing()

#-------------------------------------------------------------------------------
# Project settings (for standalone builds)
#-------------------------------------------------------------------------------

option(OPENXLA_ASYNC_BUILD_COMPILER "Build ASYNC compiler components" ON)

# Runtime components.
option(OPENXLA_ASYNC_BUILD_RUNTIME "Builds runtime components (can be individually enabled)" ON)

#-------------------------------------------------------------------------------
# Customize IREE compiler defaults.
#-------------------------------------------------------------------------------

cmake_dependent_option(IREE_BUILD_COMPILER "Builds the IREE compiler" ON ${OPENXLA_ASYNC_BUILD_COMPILER} OFF)
if(OPENXLA_NVGPU_BUILD_COMPILER)
  option(IREE_BUILD_COMPILER "Enable IREE compiler build" ON)
  option(IREE_BUILD_SAMPLES "Disable IREE samples build" OFF)
  option(IREE_TARGET_BACKEND_LLVM_CPU "Enables the 'llvm-cpu' target backend" ON)
  # Since the standalone build is primarily targeted for development mode,
  # enabled shared library builds, which enable better plugin flows
  # (but are not how we typically package for deployment).
  option(IREE_COMPILER_BUILD_SHARED_LIBS "Enables shared libraries in the compiler by default" ON)
endif()

#-------------------------------------------------------------------------------
# Customize IREE runtime defaults.
#-------------------------------------------------------------------------------

cmake_dependent_option(IREE_BUILD_RUNTIME "Builds IREE runtime components" ON ${OPENXLA_ASYNC_BUILD_RUNTIME} OFF)

#-------------------------------------------------------------------------------
# Add OpenXLA Async compiler plugin to IREE compiler tools.
#-------------------------------------------------------------------------------

set(IREE_COMPILER_PLUGIN_PATHS "${CMAKE_CURRENT_SOURCE_DIR}" CACHE STRING "async plugins")

add_subdirectory("${IREE_ROOT_DIR}" "iree_core")

# Handle various global definitions that need to be set at the global
# toolchain level.
iree_setup_toolchain()

#-------------------------------------------------------------------------------
# Absl/Tsl project dependency
#-------------------------------------------------------------------------------
if(OPENXLA_ASYNC_BUILD_RUNTIME)

message(STATUS "Fetching absl/tsl repo (this may take a few minutes)...")

include(FetchContent)

FetchContent_Declare(
  absl
  GIT_REPOSITORY https://github.com/abseil/abseil-cpp.git
  GIT_TAG 20230125.3
  GIT_SUBMODULES_RECURSE OFF
  GIT_SHALLOW OFF
  GIT_PROGRESS ON
  USES_TERMINAL_DOWNLOAD ON
)

FetchContent_Declare(
  tsl
  URL https://github.com/google/tsl/archive/refs/heads/main.zip
)

FetchContent_MakeAvailable(absl)
FetchContent_MakeAvailable(tsl)

endif()

#-------------------------------------------------------------------------------
# Async Runtime.
#
# Integration of libraries with IREE runtime via custom VM modules
#-------------------------------------------------------------------------------

if(OPENXLA_ASYNC_BUILD_RUNTIME)
  add_subdirectory(runtime)
endif()

#-------------------------------------------------------------------------------
# Google tsl dependencies
#-------------------------------------------------------------------------------

if(OPENXLA_ASYNC_BUILD_RUNTIME)
  add_subdirectory(build_tools/third_party/tsl EXCLUDE_FROM_ALL)
endif()
