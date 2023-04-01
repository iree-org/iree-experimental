// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_NVGPU_TRANSFORMS_PASSES_H_
#define OPENXLA_NVGPU_TRANSFORMS_PASSES_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace openxla::compiler::nvgpu {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createSamplePass();
} // namespace openxla::compiler::nvgpu

#endif // OPENXLA_NVGPU_TRANSFORMS_PASSES_H_
