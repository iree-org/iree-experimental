// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "openxla/compiler/nvgpu/Transforms/Passes.h"

#define GEN_PASS_DEF_SAMPLE
#include "openxla/compiler/nvgpu/Transforms/Passes.h.inc"

using namespace mlir;

namespace openxla::compiler::nvgpu {
namespace {

class SamplePass : public ::impl::SampleBase<SamplePass> {
public:
  void runOnOperation() override {}
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createSamplePass() {
  return std::make_unique<SamplePass>();
}

} // namespace openxla::compiler::nvgpu
