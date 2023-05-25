# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_library(
  NAME
    Transforms
  HDRS
    "Passes.h"
    "Passes.h.inc"
  SRCS
    "AsyncToAsyncRuntime.cpp"
  DEPS
    ::PassesIncGen
    MLIRIR
    MLIRPass
    openxla::compiler::Dialect::Async::IR
    openxla::compiler::Dialect::Async::Conversion::AsyncToRuntime
    openxla::compiler::Dialect::defs
    iree::compiler::Dialect::Util::IR
  PUBLIC
)

iree_tablegen_library(
  NAME
    PassesIncGen
  TD_FILE
    "Passes.td"
  OUTS
    --gen-pass-decls Passes.h.inc
)
