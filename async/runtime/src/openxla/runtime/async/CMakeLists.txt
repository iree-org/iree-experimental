# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

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
    async_runtime
  HDRS
    "api.h"
  SRCS
    "async_runtime.h"
    "async_runtime_cc.h"
    "async_runtime.cpp"
  DEPS
    ::defs
    tfrt::async_value
    absl::base
    absl::status
    iree::base
    iree::vm
  PUBLIC
)

iree_cc_library(
  NAME
    async_runtime_test
  HDRS
    "async_runtime_test.h"
  SRCS  
    "async_runtime_test.cpp"
  DEPS
    ::defs
    tfrt::async_value
    absl::base
    absl::status
    iree::base
    iree::vm
  PUBLIC
)

iree_cc_library(
  NAME
    async
  HDRS
    "module.h"
  TEXTUAL_HDRS
    "exports.inl"
  SRCS
    "module.c"
  DEPS
    ::defs
    ::loop_async
    ::async_runtime
    ::async_runtime_test
    iree::base
    iree::base::tracing
    iree::hal
    iree::vm
  PUBLIC
)



iree_cc_library(
  NAME
    loop_async
  HDRS
    "loop_async.h"
  SRCS
    "loop_async.c"
  DEPS
    ::defs
    iree::base
  PUBLIC
)




