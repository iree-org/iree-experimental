// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_NVGPU_CUDNN_API_H_
#define OPENXLA_RUNTIME_NVGPU_CUDNN_API_H_

#define NV_CUDNN_DISABLE_EXCEPTION

#include <cudnn_frontend.h>
#include <iree/vm/ref_cc.h>

#include <optional>

#include "iree/base/internal/span.h"
#include "iree/vm/api.h"
#include "openxla/runtime/nvgpu/dynamic_symbols.h"

namespace openxla::runtime::nvgpu {

// OpenXLA wrappers around cudnn_frontend primitives to simplify operation graph
// construction and execution. We rely on cudnn_frontend to provide C++ RAII
// wrappers for all cuDNN Graph API backend descriptors.

//===----------------------------------------------------------------------===//
// cuDNN tensor representing an abstract shaped and typed block of memory.
//===----------------------------------------------------------------------===//

class CuDNNTensor : public iree::vm::RefObject<CuDNNTensor> {
 public:
  enum class Kind { kArg, kOpResult };

  explicit CuDNNTensor(Kind kind) : kind_(kind) {}
  virtual ~CuDNNTensor() = default;

  virtual const cudnn_frontend::Tensor& tensor() const = 0;

  Kind kind() const { return kind_; }

 private:
  Kind kind_;
};

//===----------------------------------------------------------------------===//
// Tensor corresponding to the cuDNN graph arguments.
//===----------------------------------------------------------------------===//

class CuDNNArgTensor final : public CuDNNTensor {
 public:
  CuDNNArgTensor(openxla_cudnn_dynamic_symbols_t* syms,
                 cudnn_frontend::Tensor tensor);
  ~CuDNNArgTensor() override;

  const cudnn_frontend::Tensor& tensor() const override;

  static bool classof(const CuDNNTensor* tensor) {
    return tensor->kind() == Kind::kArg;
  }

 private:
  openxla_cudnn_dynamic_symbols_t* syms_;
  std::optional<cudnn_frontend::Tensor> tensor_;
};

//===----------------------------------------------------------------------===//
// Tensor corresponding to the cuDNN operation result.
//===----------------------------------------------------------------------===//

class CuDNNOpResultTensor final : public CuDNNTensor {
 public:
  CuDNNOpResultTensor(openxla_cudnn_dynamic_symbols_t* syms,
                      iree::span<CuDNNTensor* const> inputs,
                      cudnn_frontend::Operation operation,
                      cudnn_frontend::Tensor tensor);
  ~CuDNNOpResultTensor() override;

  std::vector<CuDNNTensor*> inputs() const;
  const cudnn_frontend::Operation* operation() const;
  const cudnn_frontend::Tensor& tensor() const override;

  static bool classof(const CuDNNTensor* tensor) {
    return tensor->kind() == Kind::kOpResult;
  }

 private:
  openxla_cudnn_dynamic_symbols_t* syms_;

  // Tensors inputs to the cuDNN operation. We need to keep a reference to them
  // to be able to traverse use-def chains when building an operation graph.
  std::vector<iree::vm::ref<CuDNNTensor>> inputs_;

  // cuDNN operation that computes the result tensor.
  std::optional<cudnn_frontend::Operation> operation_;
  std::optional<cudnn_frontend::Tensor> tensor_;
};

//===----------------------------------------------------------------------===//
// CuDNN operation graph.
//===----------------------------------------------------------------------===//

class CuDNNOperationGraph : public iree::vm::RefObject<CuDNNOperationGraph> {
 public:
  CuDNNOperationGraph(openxla_cudnn_dynamic_symbols_t* syms,
                      cudnn_frontend::OperationGraph graph);
  ~CuDNNOperationGraph();

  const cudnn_frontend::OperationGraph& graph() const;

 private:
  openxla_cudnn_dynamic_symbols_t* syms_;
  std::optional<cudnn_frontend::OperationGraph> graph_;
};

//===----------------------------------------------------------------------===//
// Wrappers around cuDNN APIs export from a cuDNN module to the user.
//===----------------------------------------------------------------------===//

// Creates a tensor placeholder for cuDNN graph argument.
iree::StatusOr<iree::vm::ref<CuDNNTensor>> CreateArgument(
    openxla_cudnn_dynamic_symbols_t* syms, iree::span<const int64_t> dims,
    iree::span<const int64_t> strides, int64_t uid, cudnnDataType_t dtype,
    int64_t alignment);

// Creates a pointwise relu operation.
iree::StatusOr<iree::vm::ref<CuDNNTensor>> CreatePointwiseRelu(
    openxla_cudnn_dynamic_symbols_t* syms, CuDNNTensor& input,
    double lower_clip, double upper_clip, int64_t uid, int64_t alignment);

// Creates an operation graph computing tensor results.
iree::StatusOr<iree::vm::ref<CuDNNOperationGraph>> CreateOperationGraph(
    openxla_cudnn_dynamic_symbols_t* syms, cudnnHandle_t handle,
    iree::span<CuDNNTensor* const> results);

}  // namespace openxla::runtime::nvgpu

//===----------------------------------------------------------------------===//
// Register types with IREE VM.
//===----------------------------------------------------------------------===//

IREE_VM_DECLARE_TYPE_ADAPTERS(cudnn_tensor,
                              openxla::runtime::nvgpu::CuDNNTensor);
IREE_VM_DECLARE_TYPE_ADAPTERS(cudnn_operation_graph,
                              openxla::runtime::nvgpu::CuDNNOperationGraph);

#endif  // OPENXLA_RUNTIME_NVGPU_CUDNN_API_H_
