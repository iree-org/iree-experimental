// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASYNC_OPS
#define ASYNC_OPS

include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/IR/FunctionInterfaces.td"
include "mlir/IR/OpAsmInterface.td"

include "openxla/compiler/Dialect/Async/IR/AsyncDialect.td"
include "openxla/compiler/Dialect/Async/IR/AsyncTypes.td"

//===----------------------------------------------------------------------===//
// Async op definitions
//===----------------------------------------------------------------------===//

// Base class for the operation in this dialect
class Async_Op<string mnemonic, list<Trait> traits = []> :
    Op<AsyncDialect, mnemonic, traits>;

def Async_AwaitOp : Async_Op<"await"> {
  let summary = "waits for the argument to become ready";
  let description = [{
    The `async.await` operation waits until the argument becomes ready, and for
    the `async.value` arguments it unwraps the underlying value

    Example:

    ```mlir
    %0 = ... : !async.token
    async.await %0 : !async.token

    %1 = ... : !async.value<f32>
    %2 = async.await %1 : !async.value<f32>
    ```
  }];

  let arguments = (ins Async_AnyValueOrTokenType:$operand);
  let results = (outs Optional<AnyType>:$result);

  let skipDefaultBuilders = 1;
  let hasVerifier = 1;

  let builders = [
    OpBuilder<(ins "mlir::Value":$operand,
      CArg<"mlir::ArrayRef<mlir::NamedAttribute>", "{}">:$attrs)>,
  ];

  let extraClassDeclaration = [{
    std::optional<mlir::Type> getResultType() {
      if (getResultTypes().empty()) return std::nullopt;
      return getResultTypes()[0];
    }
  }];

  let assemblyFormat = [{
    $operand `:` custom<AwaitResultType>(
      type($operand), type($result)
    ) attr-dict
  }];
}

#endif // ASYNC_OPS