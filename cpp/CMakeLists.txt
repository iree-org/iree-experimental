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
set(CMAKE_CXX_STANDARD 17)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#-------------------------------------------------------------------------------
# Core project dependency
#-------------------------------------------------------------------------------

option(IREE_SAMPLES_FETCH_SOURCE "Fetches IREE source code on-demand using CMake FetchContent" ON)
# Path to an existing IREE source directory
set(IREE_SAMPLES_SOURCE_PATH "" CACHE STRING "")

# Disable core project features not needed for these out of tree samples.
set(IREE_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
# Extend module path to find MLIR CMake modules.
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_BINARY_DIR}/lib/cmake/mlir")

if(IREE_SAMPLES_FETCH_SOURCE)
  message(STATUS "Fetching core IREE repo (this may take a few minutes)...")

  # Note: for log output, set -DFETCHCONTENT_QUIET=OFF,
  # see https://gitlab.kitware.com/cmake/cmake/-/issues/18238#note_440475
  list(APPEND CMAKE_MESSAGE_INDENT "  ")
  include(FetchContent)
  FetchContent_Declare(
    iree
    GIT_REPOSITORY https://github.com/iree-org/iree.git
    GIT_TAG 31a5dcb1910c9014db8339afd953eccaa0dc6076 # 2023-12-04
    GIT_SUBMODULES_RECURSE OFF
    GIT_SHALLOW OFF
    GIT_PROGRESS ON
    USES_TERMINAL_DOWNLOAD ON
  )
  FetchContent_MakeAvailable(iree)
  FetchContent_GetProperties(iree SOURCE_DIR IREE_SOURCE_DIR)
  list(POP_BACK CMAKE_MESSAGE_INDENT)
else()
  message(STATUS "Using IREE sources at path '${IREE_SAMPLES_SOURCE_PATH}'")
  list(APPEND CMAKE_MESSAGE_INDENT "  ")
  add_subdirectory(${IREE_SAMPLES_SOURCE_PATH} "third_party/iree" EXCLUDE_FROM_ALL)
  list(POP_BACK CMAKE_MESSAGE_INDENT)
endif()

#-------------------------------------------------------------------------------
# Common settings for dependencies
#-------------------------------------------------------------------------------

set(IMGUI_BUILD_SDL2_RENDERER_BINDING ON)
set(IMGUI_BUILD_VULKAN_BINDING ON)
set(IMGUI_BUILD_WIN32_BINDING ON)

#-------------------------------------------------------------------------------
# Individual samples
#-------------------------------------------------------------------------------

add_subdirectory(ml_overlay)
add_subdirectory(vision_inference)
add_subdirectory(vulkan_gui)
