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

StatusOr<ref<CuDNNTensor>> CuDNNArgTensor::Create(
    openxla_cudnn_dynamic_symbols_t* syms, span<const int64_t> dims,
    span<const int64_t> strides, int64_t uid, cudnnDataType_t dtype,
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

CuDNNArgTensor::CuDNNArgTensor(openxla_cudnn_dynamic_symbols_t* syms,
                               cudnn_frontend::Tensor tensor)
    : syms_(syms), tensor_(std::move(tensor)) {}

CuDNNArgTensor::~CuDNNArgTensor() {
  ScopedCuDNNStubs stubs(syms_);
  tensor_.reset();
}

const cudnn_frontend::Tensor& CuDNNArgTensor::tensor() { return *tensor_; }

}  // namespace openxla::runtime::nvgpu

//===----------------------------------------------------------------------===//
// Register types with IREE VM.
//===----------------------------------------------------------------------===//

IREE_VM_DEFINE_TYPE_ADAPTERS(cudnn_tensor,
                             openxla::runtime::nvgpu::CuDNNTensor);
