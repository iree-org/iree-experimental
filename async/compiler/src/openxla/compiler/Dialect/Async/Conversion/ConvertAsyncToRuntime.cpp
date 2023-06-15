// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/compiler/Dialect/Async/Conversion/ConvertAsyncToRuntime.h"

#include "iree/compiler/Dialect/Util/IR/UtilDialect.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "openxla/compiler/Dialect/Async/IR/Async.h"

namespace openxla::compiler::async {

using namespace mlir;
using namespace mlir::iree_compiler;

namespace {

//===----------------------------------------------------------------------===//
// AsyncAPI for importing Async VM module functions
//===----------------------------------------------------------------------===//

class AsyncAPI {
 public:
  // Import `@async.token.await` into the module
  func::FuncOp getValueAwait(PatternRewriter &rewriter, ModuleOp module);
  // Import `@async.value.query` into the module
  func::FuncOp getValueQuery(PatternRewriter &rewriter, ModuleOp module);
  // Import `@async.value.load.i32` into the module
  func::FuncOp getValueLoadI32(PatternRewriter &rewriter, ModuleOp module);
  // Import `@async.value.load.ref` into the module
  func::FuncOp getValueLoadRef(PatternRewriter &rewriter, ModuleOp module);

  SymbolTable &symTable(ModuleOp module);

  bool isScalarType(Type type) { return type.isIntOrIndexOrFloat(); }

 private:
  func::FuncOp addDecl(PatternRewriter &rewriter, ModuleOp module,
                       StringAttr name, FunctionType function_type);
  SymbolTableCollection symTable_;
};

SymbolTable &AsyncAPI::symTable(ModuleOp module) {
  return symTable_.getSymbolTable(module);
}

func::FuncOp AsyncAPI::addDecl(PatternRewriter &rewriter, ModuleOp module,
                               StringAttr name, FunctionType function_type) {
  if (auto fn = symTable_.lookupNearestSymbolFrom<func::FuncOp>(module, name))
    return fn;

  ImplicitLocOpBuilder b(UnknownLoc::get(module->getContext()), rewriter);
  b.setInsertionPointToEnd(module.getBody());

  auto fn = b.create<func::FuncOp>(name, function_type);
  fn.setPrivate();
  symTable(module).insert(fn);
  return fn;
}

func::FuncOp AsyncAPI::getValueQuery(PatternRewriter &rewriter,
                                     ModuleOp module) {
  MLIRContext *ctx = module->getContext();
  SmallVector<Type> args{ValueType::get(ctx)};
  SmallVector<Type> rets{IntegerType::get(ctx, 32)};

  auto functionType = FunctionType::get(ctx, args, rets);

  return addDecl(rewriter, module, StringAttr::get(ctx, "async.value.query"),
                 functionType);
}

func::FuncOp AsyncAPI::getValueAwait(PatternRewriter &rewriter,
                                     ModuleOp module) {
  MLIRContext *ctx = module->getContext();
  SmallVector<Type> args{ValueType::get(ctx)};
  auto functionType = FunctionType::get(ctx, args, /*rets=*/{});

  return addDecl(rewriter, module, StringAttr::get(ctx, "async.value.await"),
                 functionType);
}

func::FuncOp AsyncAPI::getValueLoadI32(PatternRewriter &rewriter,
                                       ModuleOp module) {
  MLIRContext *ctx = module->getContext();
  SmallVector<Type> args{ValueType::get(ctx)};
  SmallVector<Type> rets{IntegerType::get(ctx, 32)};
  auto functionType = FunctionType::get(ctx, args, rets);

  return addDecl(rewriter, module, StringAttr::get(ctx, "async.value.load.i32"),
                 functionType);
}

func::FuncOp AsyncAPI::getValueLoadRef(PatternRewriter &rewriter,
                                       ModuleOp module) {
  MLIRContext *ctx = module->getContext();
  SmallVector<Type> args{ValueType::get(ctx)};
  SmallVector<Type> rets{IREE::Util::ObjectType::get(ctx)};
  auto functionType = FunctionType::get(ctx, args, rets);

  return addDecl(rewriter, module, StringAttr::get(ctx, "async.value.load.ref"),
                 functionType);
}

//===----------------------------------------------------------------------===//
// Base class for all Async op conversions
//===----------------------------------------------------------------------===//

template <typename T>
struct AsyncOpConversionPattern : public OpConversionPattern<T> {
  AsyncOpConversionPattern(TypeConverter &typeConverter, MLIRContext *ctx,
                           std::shared_ptr<AsyncAPI> api)
      : OpConversionPattern<T>(typeConverter, ctx), api(std::move(api)) {}

  std::shared_ptr<AsyncAPI> api;
};

//===----------------------------------------------------------------------===//
// Lowering for `async.await` with a token operand.
//===----------------------------------------------------------------------===//

struct ConvertTokenAwaitOp : public AsyncOpConversionPattern<AwaitOp> {
  using AsyncOpConversionPattern::AsyncOpConversionPattern;

  LogicalResult matchAndRewrite(
      AwaitOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isa<TokenType>(op.getOperand().getType())) {
      return rewriter.notifyMatchFailure(op, "unsupported awaitable type");
    }
    ModuleOp module = op->getParentOfType<ModuleOp>();
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    auto awaitFuncOp = api->getValueAwait(rewriter, module);
    b.create<func::CallOp>(awaitFuncOp.getSymName(), TypeRange{},
                           adaptor.getOperands());
    auto queryFuncOp = api->getValueQuery(rewriter, module);
    auto queryOp = b.create<func::CallOp>(queryFuncOp.getSymName(),
                                          queryFuncOp.getResultTypes(),
                                          adaptor.getOperands());
    b.create<IREE::Util::StatusCheckOkOp>(queryOp.getResult(0),
                                          "failed to wait on async token");
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lowering for `async.await` with a async scalar value operand.
//===----------------------------------------------------------------------===//

struct ConvertScalarAwaitOp : public AsyncOpConversionPattern<AwaitOp> {
  using AsyncOpConversionPattern::AsyncOpConversionPattern;

  LogicalResult matchAndRewrite(
      AwaitOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isa<ValueType>(op.getOperand().getType())) {
      return rewriter.notifyMatchFailure(op, "unsupported awaitable type");
    }
    ModuleOp module = op->getParentOfType<ModuleOp>();
    auto resultType = op.getResultType();
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    if (resultType->isInteger(32)) {
      auto awaitFuncOp = api->getValueAwait(rewriter, module);
      b.create<func::CallOp>(awaitFuncOp.getSymName(), TypeRange{},
                             adaptor.getOperands());
      auto queryFuncOp = api->getValueQuery(rewriter, module);
      auto queryOp = b.create<func::CallOp>(queryFuncOp.getSymName(),
                                            queryFuncOp.getResultTypes(),
                                            adaptor.getOperands());
      b.create<IREE::Util::StatusCheckOkOp>(queryOp.getResult(0),
                                            "failed to wait on async value");
      auto loadFuncOp = api->getValueLoadI32(rewriter, module);
      rewriter.replaceOpWithNewOp<func::CallOp>(
          op, loadFuncOp.getSymName(), *resultType, adaptor.getOperands());
    } else {
      return rewriter.notifyMatchFailure(op,
                                         "unsupported awaitable scalar type");
    }

    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lowering for `async.await` with a async value of custom type operand.
//===----------------------------------------------------------------------===//

struct ConvertObjectAwaitOp : public AsyncOpConversionPattern<AwaitOp> {
  using AsyncOpConversionPattern::AsyncOpConversionPattern;

  LogicalResult matchAndRewrite(
      AwaitOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isa<ValueType>(op.getOperand().getType())) {
      return rewriter.notifyMatchFailure(op, "unsupported awaitable type");
    }
    auto resultType = op.getResultType();
    if (!resultType || api->isScalarType(*resultType)) {
      return rewriter.notifyMatchFailure(op, "unsupported async value type");
    }
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);
    auto awaitFuncOp = api->getValueAwait(rewriter, module);
    b.create<func::CallOp>(awaitFuncOp.getSymName(), TypeRange{},
                           adaptor.getOperands());
    auto queryFuncOp = api->getValueQuery(rewriter, module);
    auto queryOp = b.create<func::CallOp>(queryFuncOp.getSymName(),
                                          queryFuncOp.getResultTypes(),
                                          adaptor.getOperands());
    b.create<IREE::Util::StatusCheckOkOp>(queryOp.getResult(0),
                                          "failed to wait on async value");
    auto loadFuncOp = api->getValueLoadRef(rewriter, module);
    auto callOp = b.create<func::CallOp>(loadFuncOp.getSymName(),
                                         IREE::Util::ObjectType::get(ctx),
                                         adaptor.getOperands());
    rewriter.replaceOpWithNewOp<IREE::Util::CastOp>(op, op.getResultTypes(),
                                                    callOp.getResult(0));
    return success();
  }
};
}  // namespace

void populateAsyncToRuntimePatterns(mlir::TypeConverter &typeConverter,
                                    mlir::RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  auto api = std::make_shared<AsyncAPI>();

  patterns.insert<ConvertTokenAwaitOp>(typeConverter, ctx, api);
  patterns.insert<ConvertScalarAwaitOp>(typeConverter, ctx, api);
  patterns.insert<ConvertObjectAwaitOp>(typeConverter, ctx, api);
}

}  // namespace openxla::compiler::async
