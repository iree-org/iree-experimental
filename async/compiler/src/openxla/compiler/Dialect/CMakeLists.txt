# Copyright 2023 The OpenXLA Authors
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
    registration
  SRCS
    "PluginRegistration.cpp"
  DEPS
    ::defs
    MLIRIR
    MLIRPass
    iree::compiler::PluginAPI
    openxla::compiler::Dialect::Async::IR
    openxla::compiler::Dialect::Async::Transforms
  PUBLIC
)

iree_compiler_register_plugin(
  PLUGIN_ID
    openxla_async
  TARGET
    ::registration
)

