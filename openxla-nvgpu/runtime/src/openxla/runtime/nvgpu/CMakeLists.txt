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
    ::dynamic_symbols
    iree::runtime
  PUBLIC
)

iree_cc_library(
  NAME
    dynamic_symbols
  HDRS
    "dynamic_symbols.h"
    "status_util.h"
  TEXTUAL_HDRS
    "dynamic_symbol_tables.h"
  SRCS
    "cudnn_headers.h"
    "dynamic_symbols.c"
    "status_util.c"
  DEPS
    ::defs
    iree::base
    iree::base::core_headers
    iree::base::internal::dynamic_library
    iree::base::tracing
    # TODO: We rely on the fact that cuDNN headers are located in the same
    # directory as CUDA headers. Fix this.
    iree_cuda::headers
  PUBLIC
)