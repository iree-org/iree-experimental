// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "openxla/compiler/Dialect/Async/Conversion/ConvertAsyncToRuntime.h"
#include "openxla/compiler/Dialect/Async/IR/Async.h"
#include "openxla/compiler/Dialect/Async/Transforms/Passes.h"

#define GEN_PASS_DEF_ASYNCTOASYNCRUNTIME
#include <iostream>

#include "openxla/compiler/Dialect/Async/Transforms/Passes.h.inc"

using namespace mlir;
namespace IREE = mlir::iree_compiler::IREE;

namespace openxla::compiler::async {

namespace {

class AsyncToAsyncRuntimePass
    : public ::impl::AsyncToAsyncRuntimeBase<AsyncToAsyncRuntimePass> {
 public:
  AsyncToAsyncRuntimePass() = default;
  void runOnOperation() override;
};

}  // namespace

//===----------------------------------------------------------------------===//
void AsyncToAsyncRuntimePass::runOnOperation() {
  if (getOperation().getBody()->empty()) return;

  auto *context = &getContext();

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });
  typeConverter.addConversion(
      [](TokenType token) { return ValueType::get(token.getContext()); });
  typeConverter.addConversion(
      [](ValueType value) { return ValueType::get(value.getContext()); });

  // Ensure all async dialect operations go away.
  ConversionTarget conversionTarget(*context);
  conversionTarget.addIllegalDialect<async::AsyncDialect>();
  conversionTarget.addLegalDialect<func::FuncDialect>();
  conversionTarget.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
    return typeConverter.isSignatureLegal(op.getFunctionType()) &&
           typeConverter.isLegal(&op.getBody());
  });
  conversionTarget.addLegalDialect<IREE::Util::UtilDialect>();

  RewritePatternSet patterns(&getContext());
  populateAsyncToRuntimePatterns(typeConverter, patterns);

  if (failed(applyPartialConversion(getOperation(), conversionTarget,
                                    std::move(patterns)))) {
    getOperation().emitError() << "conversion from async to runtime failed";
    return signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createAsyncToAsyncRuntimePass() {
  return std::make_unique<AsyncToAsyncRuntimePass>();
}

}  // namespace openxla::compiler::async