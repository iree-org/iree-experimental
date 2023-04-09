// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/nvgpu/cudnn_api.h"

#include <iree/base/internal/span.h>
#include <iree/base/status.h>
#include <iree/base/status_cc.h>
#include <iree/vm/ref_cc.h>
#include <openxla/runtime/nvgpu/status_util.h>

#include <type_traits>

#include "openxla/runtime/nvgpu/cudnn_stub.h"

namespace openxla::runtime::nvgpu {

using namespace iree;

using cudnn_frontend::TensorBuilder;

// clang-format off
#include "openxla/runtime/nvgpu/cudnn_stub.h.inc"
// clang-format on

//===----------------------------------------------------------------------===//
// CuDNNArgTensor.
//===----------------------------------------------------------------------===//

CuDNNArgTensor::CuDNNArgTensor(openxla_cudnn_dynamic_symbols_t* syms,
                               cudnn_frontend::Tensor tensor)
    : CuDNNTensor(Kind::kArg), syms_(syms), tensor_(std::move(tensor)) {}

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
                                         iree::span<CuDNNTensor* const> inputs,
                                         cudnn_frontend::Operation operation,
                                         cudnn_frontend::Tensor tensor)
    : CuDNNTensor(Kind::kOpResult),
      syms_(syms),
      operation_(std::move(operation)),
      tensor_(std::move(tensor)) {
  for (CuDNNTensor* input : inputs) {
    inputs_.push_back(vm::retain_ref(input));
  }
}

CuDNNOpResultTensor::~CuDNNOpResultTensor() {
  ScopedCuDNNStubs stubs(syms_);
  operation_.reset();
  tensor_.reset();
}

std::vector<CuDNNTensor*> CuDNNOpResultTensor::inputs() const {
  std::vector<CuDNNTensor*> ptrs;
  for (auto& input : inputs_) ptrs.push_back(input.get());
  return ptrs;
}

const cudnn_frontend::Operation* CuDNNOpResultTensor::operation() const {
  return &*operation_;
}

const cudnn_frontend::Tensor& CuDNNOpResultTensor::tensor() const {
  return *tensor_;
}

//===----------------------------------------------------------------------===//
// CuDNNOperationGraph.
//===----------------------------------------------------------------------===//

CuDNNOperationGraph::CuDNNOperationGraph(openxla_cudnn_dynamic_symbols_t* syms,
                                         cudnn_frontend::OperationGraph graph)
    : syms_(syms), graph_(std::move(graph)) {}

CuDNNOperationGraph::~CuDNNOperationGraph() {
  ScopedCuDNNStubs stubs(syms_);
  graph_.reset();
}

const cudnn_frontend::OperationGraph& CuDNNOperationGraph::graph() const {
  return *graph_;
}

//===----------------------------------------------------------------------===//
// Wrappers around cuDNN APIs export from a cuDNN module to the user.
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// CreateArgument.
//===----------------------------------------------------------------------===//

StatusOr<vm::ref<CuDNNTensor>> CreateArgument(
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
  return vm::ref<CuDNNTensor>(new CuDNNArgTensor(syms, std::move(tensor)));
}

//===----------------------------------------------------------------------===//
// CreatePointwiseRelu.
//===----------------------------------------------------------------------===//

StatusOr<vm::ref<CuDNNTensor>> CreatePointwiseRelu(
    openxla_cudnn_dynamic_symbols_t* syms, CuDNNTensor& input,
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

  return vm::ref<CuDNNTensor>(new CuDNNOpResultTensor(
      syms, {&input}, std::move(operation), std::move(tensor)));
}

//===----------------------------------------------------------------------===//
// CreateOperationGraph.
//===----------------------------------------------------------------------===//

template <typename To>
static To* DynCast(CuDNNTensor* tensor) {
  return To::classof(tensor) ? static_cast<To*>(tensor) : nullptr;
}

StatusOr<vm::ref<CuDNNOperationGraph>> CreateOperationGraph(
    openxla_cudnn_dynamic_symbols_t* syms, cudnnHandle_t handle,
    span<CuDNNTensor* const> results) {
  ScopedCuDNNStubs stubs(syms);

  // Collect cuDNN operations producing tensor results.
  std::vector<CuDNNTensor*> worklist(results.begin(), results.end());
  std::vector<const cudnn_frontend::Operation*> ops;

  // TODO(ezhulenev): Take care of duplicate operations when traversing a tensor
  // use-def chains (with an end-to-end test once we'll support them).

  while (!worklist.empty()) {
    CuDNNTensor* tensor = worklist.back();
    worklist.pop_back();

    // Add cudnn_frontend operation and follow inputs.
    if (auto* op_result = DynCast<CuDNNOpResultTensor>(tensor)) {
      ops.push_back(op_result->operation());
      std::vector<CuDNNTensor*> inputs = op_result->inputs();
      worklist.insert(worklist.end(), inputs.begin(), inputs.end());
    }
  }

  // Construct a cudnn_frontend operation graph.
  auto graph = cudnn_frontend::OperationGraphBuilder()
                   .setHandle(handle)
                   .setOperationGraph(ops.size(), ops.data())
                   .build();
  IREE_RETURN_IF_ERROR(CUDNN_CONVERT_STATUS(syms, graph.get_status()));

  return vm::ref<CuDNNOperationGraph>(
      new CuDNNOperationGraph(syms, std::move(graph)));
}

}  // namespace openxla::runtime::nvgpu

//===----------------------------------------------------------------------===//
// Register types with IREE VM.
//===----------------------------------------------------------------------===//

IREE_VM_DEFINE_TYPE_ADAPTERS(cudnn_tensor,
                             openxla::runtime::nvgpu::CuDNNTensor);
IREE_VM_DEFINE_TYPE_ADAPTERS(cudnn_operation_graph,
                             openxla::runtime::nvgpu::CuDNNOperationGraph);
