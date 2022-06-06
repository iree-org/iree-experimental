# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.23)

#-------------------------------------------------------------------------------
# Project configuration
#-------------------------------------------------------------------------------

project(iree-samples C CXX)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#-------------------------------------------------------------------------------
# Core project dependency
#-------------------------------------------------------------------------------

message(STATUS "Fetching core IREE repo (this may take a few minutes)...")

include(FetchContent)

FetchContent_Declare(
  iree
  GIT_REPOSITORY https://github.com/google/iree.git
  GIT_TAG d9f6498ca22a03ede66bc82e20d733faaa768be4 # 2022-06-06
  GIT_SUBMODULES_RECURSE OFF
  GIT_SHALLOW ON
  GIT_PROGRESS ON
)

# Extend module path to find MLIR CMake modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/mlir")

# Disable core project features not needed for these out of tree samples.
set(IREE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(iree)
FetchContent_GetProperties(iree SOURCE_DIR IREE_SOURCE)

#-------------------------------------------------------------------------------
# Individual samples
#-------------------------------------------------------------------------------

add_subdirectory(vulkan_gui)
