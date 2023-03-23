//===- CUDNNDialect.cpp - CUDNN dialect ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CUDNN/IR/CUDNNDialect.h"
#include "CUDNN/IR/CUDNNOps.h"
#include "CUDNN/IR/CUDNNTypes.h"

using namespace mlir;
using namespace mlir::cudnn;

#include "CUDNN/IR/CUDNNOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// CUDNN dialect.
//===----------------------------------------------------------------------===//

void CUDNNDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "CUDNN/IR/CUDNNOps.cpp.inc"
      >();
  registerTypes();
}
