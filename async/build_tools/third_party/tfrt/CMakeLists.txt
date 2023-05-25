# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TFRT_ROOT "${OPENXLA_ASYNC_ROOT_DIR}/third_party/tfrt/")
set(TFRT_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}")

external_cc_library(
  PACKAGE
    tfrt
  NAME
    ref_count
  ROOT
    ${TFRT_ROOT}
  INCLUDES
    "${TFRT_ROOT}/include"
    "${TFRT_BINARY_DIR}/include"
  HDRS
    "include/tfrt/concurrency/ref_count.h"
  COPTS
    -fno-rtti
    -fno-exceptions
  PUBLIC
)

external_cc_library(
  PACKAGE
    tfrt
  NAME
    concurrent_vector
  ROOT
    ${TFRT_ROOT}
  INCLUDES
    "${TFRT_ROOT}/include"
    "${TFRT_BINARY_DIR}/include"
  HDRS
    "include/tfrt/concurrency/concurrent_vector.h"
  DEPS
    absl::base
    absl::status
  COPTS
    -fno-rtti
    -fno-exceptions
  PUBLIC
)

external_cc_library(
  PACKAGE
    tfrt
  NAME
    support
  ROOT
    ${TFRT_ROOT}
  INCLUDES
    "${TFRT_ROOT}/include"
    "${TFRT_BINARY_DIR}/include"
  HDRS
    "include/tfrt/support/alloc.h"
    "include/tfrt/support/logging.h"
  SRCS
    "lib/support/alloc.cc"
    "lib/support/logging.cc"
  DEPS
    LLVMSupport
  COPTS
    -fno-rtti
    -fno-exceptions
  PUBLIC
)


external_cc_library(
  PACKAGE
    tfrt
  NAME
    async_value
  ROOT
    ${TFRT_ROOT}
  INCLUDES
    "${TFRT_ROOT}/include"
    "${TFRT_BINARY_DIR}/include"
  HDRS
    "include/tfrt/concurrency/async_value.h"
    "include/tfrt/concurrency/async_value_ref.h"
    "include/tfrt/concurrency/chain.h"
  SRCS
    "lib/concurrency/async_value.cc"
    "lib/concurrency/async_value_ref.cc"
  DEPS
    LLVMSupport
    tfrt::concurrent_vector
    tfrt::ref_count
    tfrt::support
    absl::base
    absl::status
  COPTS
    -fno-rtti
    -fno-exceptions
  PUBLIC
)