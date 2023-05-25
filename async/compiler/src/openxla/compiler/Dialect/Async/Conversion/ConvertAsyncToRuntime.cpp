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
namespace IREE = mlir::iree_compiler::IREE;

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// AsyncAPI for importing Async VM module functions
//===----------------------------------------------------------------------===//

class AsyncAPI {
 public:
  // Imports `@async.token.await` into the module
  func::FuncOp getTokenAwait(PatternRewriter &rewriter, ModuleOp module);
  // Imports `@async.value.await.i32` into the module
  func::FuncOp getValueAwaitI32(PatternRewriter &rewriter, ModuleOp module);
  // Imports `@async.value.await.ref` into the module
  func::FuncOp getValueAwaitRef(PatternRewriter &rewriter, ModuleOp module);

  SymbolTable &symTable(ModuleOp module);

  bool isScalarType(Type type) {
    return type.isInteger(32) || type.isInteger(64) || type.isF32() ||
           type.isF64();
  }

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

func::FuncOp AsyncAPI::getTokenAwait(PatternRewriter &rewriter,
                                     ModuleOp module) {
  MLIRContext *ctx = module->getContext();
  SmallVector<Type> args{ValueType::get(ctx)};
  auto functionType = FunctionType::get(ctx, args, /*rets=*/{});

  return addDecl(rewriter, module,
                 StringAttr::get(ctx, "async.value.await.token"), functionType);
}

func::FuncOp AsyncAPI::getValueAwaitI32(PatternRewriter &rewriter,
                                        ModuleOp module) {
  MLIRContext *ctx = module->getContext();
  SmallVector<Type> args{ValueType::get(ctx)};
  SmallVector<Type> rets{IntegerType::get(ctx, 32)};
  auto functionType = FunctionType::get(ctx, args, rets);

  return addDecl(rewriter, module,
                 StringAttr::get(ctx, "async.value.await.i32"), functionType);
}

func::FuncOp AsyncAPI::getValueAwaitRef(PatternRewriter &rewriter,
                                        ModuleOp module) {
  MLIRContext *ctx = module->getContext();
  SmallVector<Type> args{ValueType::get(ctx)};
  SmallVector<Type> rets{IREE::Util::ObjectType::get(ctx)};
  auto functionType = FunctionType::get(ctx, args, rets);

  return addDecl(rewriter, module,
                 StringAttr::get(ctx, "async.value.await.ref"), functionType);
}

class FuncOpConversion : public OpConversionPattern<func::FuncOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      func::FuncOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    FunctionType type = op.getFunctionType();
    // Convert the original function arguments.
    TypeConverter::SignatureConversion result(type.getNumInputs());
    for (unsigned i = 0; i < type.getNumInputs(); ++i) {
      if (failed(getTypeConverter()->convertSignatureArg(i, type.getInput(i),
                                                         result))) {
        return failure();
      }
    }
    // Convert the original function results.
    SmallVector<Type, 1> converted_results;
    if (failed(getTypeConverter()->convertTypes(type.getResults(),
                                                converted_results))) {
      return failure();
    }
    if (failed(rewriter.convertRegionTypes(&op.getBody(), *getTypeConverter(),
                                           &result))) {
      return failure();
    }

    // Update the function signature.
    rewriter.updateRootInPlace(op, [&] {
      op.setType(FunctionType::get(op.getContext(), result.getConvertedTypes(),
                                   converted_results));
    });

    return success();
  }
};

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

struct ConvertTokenOp : public AsyncOpConversionPattern<AwaitOp> {
  using AsyncOpConversionPattern::AsyncOpConversionPattern;

  LogicalResult matchAndRewrite(
      AwaitOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (!isa<TokenType>(op.getOperand().getType())) {
      return rewriter.notifyMatchFailure(op, "unsupported awaitable type");
    }
    ModuleOp module = op->getParentOfType<ModuleOp>();
    ImplicitLocOpBuilder b(op->getLoc(), rewriter);

    auto funcOp = api->getTokenAwait(rewriter, module);
    rewriter.replaceOpWithNewOp<func::CallOp>(
        op, funcOp.getSymName(), TypeRange{}, adaptor.getOperands());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Lowering for `async.await` with a async scalar value operand.
//===----------------------------------------------------------------------===//

struct ConvertValueScalarOp : public AsyncOpConversionPattern<AwaitOp> {
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
      auto funcOp = api->getValueAwaitI32(rewriter, module);
      // auto callOp = b.create<func::CallOp>(funcOp.getSymName(), *resultType,
      //                                      adaptor.getOperands());
      // rewriter.replaceOp(op, callOp.getResult(0));
      rewriter.replaceOpWithNewOp<func::CallOp>(
          op, funcOp.getSymName(), *resultType, adaptor.getOperands());
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

struct ConvertValueRefOp : public AsyncOpConversionPattern<AwaitOp> {
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
    auto funcOp = api->getValueAwaitRef(rewriter, module);
    auto callOp = b.create<func::CallOp>(funcOp.getSymName(),
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

  patterns.insert<FuncOpConversion>(typeConverter, ctx);
  patterns.insert<ConvertTokenOp>(typeConverter, ctx, api);
  patterns.insert<ConvertValueScalarOp>(typeConverter, ctx, api);
  patterns.insert<ConvertValueRefOp>(typeConverter, ctx, api);
}

}  // namespace openxla::compiler::async