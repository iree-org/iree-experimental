# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

add_subdirectory(test)

iree_cc_library(
  NAME
    defs
  INCLUDES
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../../..>"
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/../../..>"
  PUBLIC
)

iree_cc_library(
  NAME
    cudnn_module
  HDRS
    "cudnn_module.h"
  SRCS
    "cudnn_module.cpp"
  DEPS
    ::defs
    iree::runtime
  PUBLIC
)
