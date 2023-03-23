//===- CUDNNTypes.cpp - CUDNN dialect types -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "CUDNN/IR/CUDNNTypes.h"

#include "CUDNN/IR/CUDNNDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::cudnn;

static LogicalResult
parseDimensionList(AsmParser &parser,
                   FailureOr<::llvm::SmallVector<int64_t>> &dims,
                   FailureOr<Type> &type) {
  llvm::SmallVector<int64_t> res;
  Type resT;
  if (!succeeded(parser.parseDimensionList(res, /*allowDynamic=*/true,
                                           /*withTrailingX=*/true)) ||
      !succeeded(parser.parseType(resT)))
    return failure();
  dims.emplace(std::move(res));
  type.emplace(std::move(resT));
  return success();
}

static void printDimensionList(AsmPrinter &printer, ArrayRef<int64_t> dims,
                               Type type) {
  llvm::interleave(dims, printer.getStream(), "x");
  printer << 'x' << type;
}

#define GET_TYPEDEF_CLASSES
#include "CUDNN/IR/CUDNNTypes.cpp.inc"

void CUDNNDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "CUDNN/IR/CUDNNTypes.cpp.inc"
      >();
}
