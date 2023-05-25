# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_add_all_subdirs()

iree_cc_library(
  NAME
    IR
  HDRS
    "Async.h"
  TEXTUAL_HDRS
    "AsyncOps.h.inc"
    "AsyncOps.cpp.inc"
    "AsyncDialect.cpp.inc"
    "AsyncDialect.h.inc"
    "AsyncTypes.cpp.inc"
    "AsyncTypes.h.inc"
  SRCS
    "Async.cpp"
  DEPS
    ::AsyncOpsGen
    ::AsyncTypesGen
    LLVMSupport
    MLIRFuncDialect
    MLIRIR
    MLIRSupport
    openxla::compiler::Dialect::defs
    iree::compiler::Dialect::Util::IR
  PUBLIC
)

iree_tablegen_library(
  NAME
    AsyncOpsGen
  TD_FILE
    "AsyncOps.td"
  OUTS
    --gen-dialect-decls AsyncDialect.h.inc
    --gen-dialect-defs AsyncDialect.cpp.inc
    --gen-op-decls AsyncOps.h.inc
    --gen-op-defs AsyncOps.cpp.inc
)

iree_tablegen_library(
  NAME
    AsyncTypesGen
  TD_FILE
    "AsyncTypes.td"
  OUTS
    --gen-typedef-decls AsyncTypes.h.inc
    --gen-typedef-defs AsyncTypes.cpp.inc
)

iree_tablegen_doc(
  NAME
    AsyncDialectDocGen
  TD_FILE
    "AsyncOps.td"
  OUTS
    --gen-dialect-doc -dialect=async src/async/IR/AsyncDialect.md
)