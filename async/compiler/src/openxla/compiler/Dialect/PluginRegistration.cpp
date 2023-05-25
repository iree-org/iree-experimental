// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/PluginAPI/Client.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "openxla/compiler/Dialect/Async/IR/Async.h"
#include "openxla/compiler/Dialect/Async/Transforms/Passes.h"

using namespace mlir;
using namespace mlir::iree_compiler;

namespace detail {
namespace {

#define GEN_PASS_REGISTRATION
#include "openxla/compiler/Dialect/Async/Transforms/Passes.h.inc"

}  // namespace
}  // namespace detail

namespace {

struct AsyncOptions {
  void bindOptions(OptionsBinder &binder) {}
};

struct AsyncSession : public PluginSession<AsyncSession, AsyncOptions> {
  static void registerPasses() { ::detail::registerPasses(); }

  void onRegisterDialects(DialectRegistry &registry) override {
    registry.insert<openxla::compiler::async::AsyncDialect>();
  }

  void extendPreprocessingPassPipeline(OpPassManager &pm) override {
    pm.addPass(openxla::compiler::async::createAsyncToAsyncRuntimePass());
  }
};

}  // namespace

IREE_DEFINE_COMPILER_OPTION_FLAGS(AsyncOptions);

extern "C" bool iree_register_compiler_plugin_openxla_async(
    mlir::iree_compiler::PluginRegistrar *registrar) {
  registrar->registerPlugin<AsyncSession>("openxla-async");
  return true;
}
