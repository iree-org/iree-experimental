//===- CUDNNTypes.cpp - CUDNN dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace openxla::compiler::nvgpu::cudnn;

static LogicalResult parseDimensionList(AsmParser &parser,
                                        ::llvm::SmallVector<int64_t> &dims,
                                        Type &type) {
  llvm::SmallVector<int64_t> res;
  Type resT;
  if (!succeeded(parser.parseDimensionList(res, /*allowDynamic=*/true,
                                           /*withTrailingX=*/true)) ||
      !succeeded(parser.parseType(resT)))
    return failure();
  dims = std::move(res);
  type = std::move(resT);
  return success();
}

static void printDimensionList(AsmPrinter &printer, ArrayRef<int64_t> dims,
                               Type type) {
  llvm::interleave(dims, printer.getStream(), "x");
  printer << 'x' << type;
}

#define GET_TYPEDEF_CLASSES
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.cpp.inc"

void CUDNNDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.cpp.inc"
      >();
}
