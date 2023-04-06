// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>
#include <algorithm>

#include "mhlo/IR/hlo_ops.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.h"
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNOps.h"
#include "openxla/compiler/nvgpu/Transforms/Passes.h"

#define GEN_PASS_DEF_CONVERTMHLOTOCUDNN
#include "openxla/compiler/nvgpu/Transforms/Passes.h.inc"

using namespace mlir;

namespace openxla::compiler::nvgpu {

static std::optional<APFloat> getFloatSplatValue(Value value) {
  auto op = value.getDefiningOp<mhlo::ConstantOp>();
  if (!op) return std::nullopt;
  ElementsAttr attr = op.getValue();
  if (!attr.isSplat()) return std::nullopt;
  if (!attr.getElementType().isa<FloatType>()) return std::nullopt;
  return attr.getSplatValue<APFloat>();
}

static SmallVector<int64_t> getRowMajorStrides(ArrayRef<int64_t> shape) {
  SmallVector<int64_t> strides(shape.size(), 1);
  std::partial_sum(shape.rbegin(),
                   shape.rend() - 1,
                   strides.rbegin() + 1,
                   std::multiplies<int64_t>());
  return strides;
}

static cudnn::TensorDescType getTensorDescType(TensorType tensor_type) {
  auto shape = tensor_type.getShape();
  Type element_type = tensor_type.getElementType();
  int alignment = 0;
  auto strides = getRowMajorStrides(shape);
  return cudnn::TensorDescType::get(tensor_type.getContext(),
                                    shape, element_type, alignment, strides);
}

namespace {

struct ConvertClamp : public OpRewritePattern<mhlo::ClampOp> {
  using OpRewritePattern<mhlo::ClampOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mhlo::ClampOp op,
                                PatternRewriter& rewriter) const override {
    auto min = getFloatSplatValue(op.getMin());
    auto max = getFloatSplatValue(op.getMax());
    if (!min || !max || !max->isNaN())
      return rewriter.notifyMatchFailure(op, "unexpected bounds");
    TensorType tensor_type = op.getOperand().getType();
    cudnn::TensorDescType tensor_desc_type = getTensorDescType(tensor_type);
    Value input = rewriter.create<UnrealizedConversionCastOp>(
        op.getLoc(), tensor_desc_type, op.getOperand()).getResult(0);
    Value result = rewriter.create<cudnn::PointWiseReluOp>(
        op.getLoc(), tensor_desc_type, input, tensor_type.getElementType(),
        APFloat(min->convertToDouble()));
    rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
        op, tensor_type, result);
    return success();
  }
};

class ConvertMHLOToCUDNN : public ::impl::ConvertMHLOToCUDNNBase<
    ConvertMHLOToCUDNN> {
 public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.insert<ConvertClamp>(&getContext());
    if (failed(::applyPatternsAndFoldGreedily(getOperation(),
                                              std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertMHLOToCUDNNPass() {
  return std::make_unique<ConvertMHLOToCUDNN>();
}

} // namespace openxla::compiler::nvgpu
