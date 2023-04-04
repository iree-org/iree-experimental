//===- CUDNNOps.cpp - CUDNN dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNOps.h"
#include "mlir/IR/OpImplementation.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h"

#define GET_OP_CLASSES
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNOps.cpp.inc"
