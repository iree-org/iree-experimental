// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef OPENXLA_ASYNC_TRANSFORMS_PASSES_H_
#define OPENXLA_ASYNC_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace openxla::compiler::async {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAsyncToAsyncRuntimePass();
}  // namespace openxla::compiler::async

#endif  // OPENXLA_ASYNC_TRANSFORMS_PASSES_H_
