// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/Dialect/Async/IR/Async.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/IRMapping.h"

using namespace mlir;
using namespace openxla::compiler::async;

#include "openxla/compiler/Dialect/Async/IR/AsyncDialect.cpp.inc"

void AsyncDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "openxla/compiler/Dialect/Async/IR/AsyncOps.cpp.inc"
      >();
  addTypes<
#define GET_TYPEDEF_LIST
#include "openxla/compiler/Dialect/Async/IR/AsyncTypes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
/// AwaitOp
//===----------------------------------------------------------------------===//

void AwaitOp::build(OpBuilder &builder, OperationState &result, Value operand,
                    ArrayRef<NamedAttribute> attrs) {
  result.addOperands({operand});
  result.attributes.append(attrs.begin(), attrs.end());

  // Add unwrapped async.value type to the returned values types.
  if (auto valueType = operand.getType().dyn_cast<ValueType>())
    result.addTypes(valueType.getValueType());
}

static ParseResult parseAwaitResultType(OpAsmParser &parser, Type &operandType,
                                        Type &resultType) {
  if (parser.parseType(operandType)) return failure();

  // Add unwrapped async.value type to the returned values types.
  if (auto valueType = operandType.dyn_cast<ValueType>())
    resultType = valueType.getValueType();

  return success();
}

static void printAwaitResultType(OpAsmPrinter &p, Operation *op,
                                 Type operandType, Type resultType) {
  p << operandType;
}

LogicalResult AwaitOp::verify() {
  Type argType = getOperand().getType();

  // Awaiting on a token does not have any results.
  if (argType.isa<TokenType>() && !getResultTypes().empty())
    return emitOpError("awaiting on a token must have empty result");

  // Awaiting on a value unwraps the async value type.
  if (auto value = argType.dyn_cast<ValueType>()) {
    if (*getResultType() != value.getValueType())
      return emitOpError() << "result type " << *getResultType()
                           << " does not match async value type "
                           << value.getValueType();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "openxla/compiler/Dialect/Async/IR/AsyncOps.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd type method definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "openxla/compiler/Dialect/Async/IR/AsyncTypes.cpp.inc"

void ValueType::print(AsmPrinter &printer) const {
  // Opaque async value type (`!async.value`)
  if (!getValueType()) return;

  printer << "<";
  printer.printType(getValueType());
  printer << '>';
}

Type ValueType::parse(mlir::AsmParser &parser) {
  // Opaque async value type (`async.value`)
  if (failed(parser.parseOptionalLess())) {
    return ValueType::get(parser.getContext());
  }

  Type ty;
  if (parser.parseType(ty) || parser.parseGreater()) {
    parser.emitError(parser.getNameLoc(), "failed to parse async value type");
    return Type();
  }
  return ValueType::get(ty);
}

bool ValueType::isOpaque() { return !getValueType(); }