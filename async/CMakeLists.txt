# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cmake_minimum_required(VERSION 3.21...3.24)

project(ASYNC)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

set(OPENXLA_ASYNC_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# TODO: Fix this once the project is slotted into place.
if(NOT IREE_ROOT_DIR)
  set(IREE_ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../iree")
endif()

set(IREE_CXX_STANDARD ${CMAKE_CXX_STANDARD})

# Customize defaults.
option(IREE_BUILD_COMPILER "Disable compiler for runtime-library build" ON)
option(IREE_BUILD_SAMPLES "Disable samples for runtime-library build" OFF)
option(IREE_COMPILER_BUILD_SHARED_LIBS "Enables shared libraries in the compiler by default" ON)

set(IREE_COMPILER_PLUGIN_PATHS "${CMAKE_CURRENT_SOURCE_DIR}" CACHE STRING "async plugins")
add_subdirectory("${IREE_ROOT_DIR}" "iree_core")

# Handle various global definitions that need to be set at the global
# toolchain level.
iree_setup_toolchain()

#-------------------------------------------------------------------------------
# Async Runtime.
#
#-------------------------------------------------------------------------------

add_subdirectory(runtime)
add_subdirectory(tools)

#-------------------------------------------------------------------------------
# absl dependencies
#-------------------------------------------------------------------------------

add_subdirectory(third_party/abseil-cpp)


#-------------------------------------------------------------------------------
# tfrt dependencies
#-------------------------------------------------------------------------------

add_subdirectory(build_tools/third_party/tfrt EXCLUDE_FROM_ALL)

