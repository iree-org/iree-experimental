// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/nvgpu/cudnn_tensor.h"

#include <iree/base/internal/span.h>
#include <iree/base/status.h>
#include <iree/base/status_cc.h>
#include <iree/vm/ref_cc.h>
#include <openxla/runtime/nvgpu/status_util.h>

#include "openxla/runtime/nvgpu/cudnn_stub.h"

namespace openxla::runtime::nvgpu {

using cudnn_frontend::TensorBuilder;

using iree::span;
using iree::StatusOr;
using iree::vm::ref;

// clang-format off
#include "openxla/runtime/nvgpu/cudnn_stub.h.inc"
// clang-format on

//===----------------------------------------------------------------------===//
// CuDNNArgTensor.
//===----------------------------------------------------------------------===//

CuDNNArgTensor::CuDNNArgTensor(openxla_cudnn_dynamic_symbols_t* syms,
                               cudnn_frontend::Tensor tensor)
    : syms_(syms), tensor_(std::move(tensor)) {}

CuDNNArgTensor::~CuDNNArgTensor() {
  ScopedCuDNNStubs stubs(syms_);
  tensor_.reset();
}

const cudnn_frontend::Tensor& CuDNNArgTensor::tensor() const {
  return *tensor_;
}

//===----------------------------------------------------------------------===//
// CuDNNOpResultTensor.
//===----------------------------------------------------------------------===//

CuDNNOpResultTensor::CuDNNOpResultTensor(openxla_cudnn_dynamic_symbols_t* syms,
                                         cudnn_frontend::Operation operation,
                                         cudnn_frontend::Tensor tensor)
    : syms_(syms),
      operation_(std::move(operation)),
      tensor_(std::move(tensor)) {}

CuDNNOpResultTensor::~CuDNNOpResultTensor() {
  ScopedCuDNNStubs stubs(syms_);
  operation_.reset();
  tensor_.reset();
}

const cudnn_frontend::Tensor& CuDNNOpResultTensor::tensor() const {
  return *tensor_;
}

//===----------------------------------------------------------------------===//
// Wrappers around cuDNN APIs export from a cuDNN module to the user.
//===----------------------------------------------------------------------===//

StatusOr<ref<CuDNNTensor>> CreateArgument(openxla_cudnn_dynamic_symbols_t* syms,
                                          span<const int64_t> dims,
                                          span<const int64_t> strides,
                                          int64_t uid, cudnnDataType_t dtype,
                                          int64_t alignment) {
  ScopedCuDNNStubs stubs(syms);
  cudnn_frontend::Tensor tensor = cudnn_frontend::TensorBuilder()
                                      .setDim(dims.size(), dims.data())
                                      .setStride(strides.size(), strides.data())
                                      .setId(uid)
                                      .setAlignment(alignment)
                                      .setDataType(dtype)
                                      .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, tensor.get_status()));
  return ref<CuDNNTensor>(new CuDNNArgTensor(syms, std::move(tensor)));
}

StatusOr<ref<CuDNNTensor>> CreatePointwiseRelu(
    openxla_cudnn_dynamic_symbols_t* syms, const CuDNNTensor& input,
    double lower_clip, double upper_clip, int64_t uid, int64_t alignment) {
  ScopedCuDNNStubs stubs(syms);

  // Prepare tensor descriptor for activation output.
  cudnn_frontend::Tensor tensor = cudnn_frontend::TensorBuilder()
                                      .cloneFrom(input.tensor(), uid)
                                      .setAlignment(alignment)
                                      .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, tensor.get_status()));

  // Prepare activation descriptor.
  cudnn_frontend::PointWiseDesc activation =
      cudnn_frontend::PointWiseDescBuilder()
          .setMode(CUDNN_POINTWISE_RELU_FWD)
          .setClipping(lower_clip, upper_clip)
          .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, activation.get_status()));

  // Create operation.
  cudnn_frontend::Operation operation =
      cudnn_frontend::OperationBuilder(
          CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR)
          .setxDesc(input.tensor())
          .setyDesc(tensor)
          .setpwDesc(activation)
          .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, operation.get_status()));

  return ref<CuDNNTensor>(
      new CuDNNOpResultTensor(syms, std::move(operation), std::move(tensor)));
}

}  // namespace openxla::runtime::nvgpu

//===----------------------------------------------------------------------===//
// Register types with IREE VM.
//===----------------------------------------------------------------------===//

IREE_VM_DEFINE_TYPE_ADAPTERS(cudnn_tensor,
                             openxla::runtime::nvgpu::CuDNNTensor);
