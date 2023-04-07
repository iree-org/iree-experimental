// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_NVGPU_CUDNN_TENSOR_H_
#define OPENXLA_RUNTIME_NVGPU_CUDNN_TENSOR_H_

#define NV_CUDNN_DISABLE_EXCEPTION

#include <cudnn_frontend.h>
#include <iree/vm/ref_cc.h>

#include <optional>

#include "iree/base/internal/span.h"
#include "iree/vm/api.h"
#include "openxla/runtime/nvgpu/dynamic_symbols.h"

namespace openxla::runtime::nvgpu {

//===----------------------------------------------------------------------===//
// cuDNN tensor representing an abstract shaped and typed block of memory.
//===----------------------------------------------------------------------===//

class CuDNNTensor : public iree::vm::RefObject<CuDNNTensor> {
 public:
  virtual ~CuDNNTensor() = default;
};

// TODO: We need a custom RTTI support for vm::ref data types to be able to
// conveniently cast and dyn_cast to various tensor types.

//===----------------------------------------------------------------------===//
// Tensor corresponding to the cuDNN graph arguments.
//===----------------------------------------------------------------------===//

class CuDNNArgTensor final : public CuDNNTensor {
 public:
  static iree::StatusOr<iree::vm::ref<CuDNNTensor>> Create(
      openxla_cudnn_dynamic_symbols_t* syms, iree::span<const int64_t> dims,
      iree::span<const int64_t> strides, int64_t uid, cudnnDataType_t dtype,
      int64_t alignment);

  const cudnn_frontend::Tensor& tensor();

 private:
  CuDNNArgTensor(openxla_cudnn_dynamic_symbols_t* syms,
                 cudnn_frontend::Tensor tensor);
  ~CuDNNArgTensor() override;

  openxla_cudnn_dynamic_symbols_t* syms_;
  std::optional<cudnn_frontend::Tensor> tensor_;
};

//===----------------------------------------------------------------------===//
// Tensor corresponding to the cuDNN operation result.
//===----------------------------------------------------------------------===//

// TODO: Tensor can be a result of cuDNN operation (think of MLIR BlockArgument
// vs Value). CuDNNOpResultTensor has to carry tensor itself, and the operation
// that produced it.

}  // namespace openxla::runtime::nvgpu

//===----------------------------------------------------------------------===//
// Register types with IREE VM.
//===----------------------------------------------------------------------===//

IREE_VM_DECLARE_TYPE_ADAPTERS(cudnn_tensor,
                              openxla::runtime::nvgpu::CuDNNTensor);

#endif  // OPENXLA_RUNTIME_NVGPU_CUDNN_TENSOR_H_
