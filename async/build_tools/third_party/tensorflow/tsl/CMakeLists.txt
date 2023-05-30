# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(TENSORFLOW_ROOT "${OPENXLA_ASYNC_ROOT_DIR}/third_party/tensorflow/")
set(EIGEN_ROOT "${OPENXLA_ASYNC_ROOT_DIR}/third_party/eigen/")
set(TENSORFLOW_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/../../")

add_subdirectory(protobuf)

external_cc_library(
  PACKAGE
    tsl
  NAME
    ref_count
  ROOT
    ${TENSORFLOW_ROOT}
  INCLUDES
    "${TENSORFLOW_ROOT}/"
    "${TENSORFLOW_BINARY_DIR}/"
  HDRS
    "tensorflow/tsl/concurrency/ref_count.h"
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
    ${TENSORFLOW_ROOT}
  INCLUDES
    "${TENSORFLOW_ROOT}/"
    "${TENSORFLOW_BINARY_DIR}/"
  HDRS
    "tensorflow/tsl/concurrency/concurrent_vector.h"
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
    ${TENSORFLOW_ROOT}
  INCLUDES
    "${TENSORFLOW_ROOT}/"
    "${TENSORFLOW_BINARY_DIR}/"
  HDRS
    "tensorflow/tsl/concurrency/async_value.h"
    "tensorflow/tsl/concurrency/async_value_ref.h"
    "tensorflow/tsl/concurrency/chain.h"
  SRCS
    "tensorflow/tsl/concurrency/async_value.cc"
    "tensorflow/tsl/concurrency/async_value_ref.cc"
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

external_cc_library(
  PACKAGE
    tsl
  NAME
    threadpool
  ROOT
    ${TENSORFLOW_ROOT}
  INCLUDES
    "${TENSORFLOW_ROOT}/"
    "${EIGEN_ROOT}/"
    "${TENSORFLOW_BINARY_DIR}/"
  HDRS
    "tensorflow/tsl/platform/threadpool.h"
  SRCS
    "tensorflow/tsl/platform/default/mutex.cc"
    "tensorflow/tsl/platform/threadpool.cc"
  DEPS
    absl::base
    absl::status
  COPTS
    -fno-rtti
    -fno-exceptions
  PUBLIC
)