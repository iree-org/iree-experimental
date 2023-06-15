# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TSL_ROOT "${OPENXLA_ASYNC_ROOT_DIR}/third_party/tsl/")
set(TSL_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../")

external_cc_library(
  PACKAGE
    tsl
  NAME
    ref_count
  ROOT
    ${TSL_ROOT}
  INCLUDES
    "${TSL_ROOT}/"
    "${TSL_BINARY_DIR}/"
  HDRS
    "tsl/concurrency/ref_count.h"
  COPTS
    -fno-rtti
    -fno-exceptions
  PUBLIC
)

external_cc_library(
  PACKAGE
    tsl
  NAME
    concurrent_vector
  ROOT
    ${TSL_ROOT}
  INCLUDES
    "${TSL_ROOT}/"
    "${TSL_BINARY_DIR}/"
  HDRS
    "tsl/concurrency/concurrent_vector.h"
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
    tsl
  NAME
    async_value
  ROOT
    ${TSL_ROOT}
  INCLUDES
    "${TSL_ROOT}/"
    "${TSL_BINARY_DIR}/"
  HDRS
    "tsl/concurrency/async_value.h"
    "tsl/concurrency/async_value_ref.h"
    "tsl/concurrency/chain.h"
  SRCS
    "tsl/concurrency/async_value.cc"
    "tsl/concurrency/async_value_ref.cc"
  DEPS
    LLVMSupport
    tsl::concurrent_vector
    tsl::ref_count
    absl::base
    absl::status
  COPTS
    -fno-rtti
    -fno-exceptions
  PUBLIC
)
