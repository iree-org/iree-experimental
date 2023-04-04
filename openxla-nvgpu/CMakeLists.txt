# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.24)

project(OPENXLA_NVGPU)

# TODO: Fix this once the project is slotted into place.
if(NOT IREE_ROOT_DIR)
  set(IREE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../iree")
endif()

# Customize defaults.
option(IREE_BUILD_COMPILER "Disable compiler for runtime-library build" ON)
option(IREE_BUILD_SAMPLES "Disable samples for runtime-library build" OFF)
option(IREE_HAL_DRIVER_DEFAULTS "Disable all HAL drivers by default" OFF)
option(IREE_HAL_DRIVER_CUDA "Enables the 'cuda' runtime HAL driver" ON)
option(IREE_TARGET_BACKEND_DEFAULTS "Disables target backend" OFF)
option(IREE_TARGET_BACKEND_CUDA "Enables CUDA target backend" ON)
option(IREE_COMPILER_BUILD_SHARED_LIBS "Enables shared libraries in the compiler by default" ON)

set(IREE_COMPILER_PLUGIN_PATHS "${CMAKE_CURRENT_SOURCE_DIR}" CACHE STRING "OpenXLA nvgpu plugins")
add_subdirectory("${IREE_ROOT_DIR}" "iree_core")
