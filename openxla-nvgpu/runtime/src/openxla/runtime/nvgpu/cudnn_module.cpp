// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/nvgpu/cudnn_module.h"

#include <iree/base/status.h>
#include <iree/base/status_cc.h>
#include <iree/vm/list.h>
#include <iree/vm/ref_cc.h>
#include <openxla/runtime/nvgpu/cudnn_api.h>

#include <cstdint>
#include <cstdio>
#include <memory>

#include "iree/hal/drivers/cuda/cuda_device.h"
#include "iree/modules/hal/types.h"
#include "iree/vm/native_module_cc.h"
#include "openxla/runtime/nvgpu/dynamic_symbols.h"
#include "openxla/runtime/nvgpu/status_util.h"

namespace openxla::runtime::nvgpu {

using namespace iree;

//===----------------------------------------------------------------------===//
// CuDNN module state encapsulates all the state required for running cuDNN
// operations (launching cuDNN graphs on a stream) at run time.
//===----------------------------------------------------------------------===//

class CuDNNModuleState {
 public:
  CuDNNModuleState(openxla_cudnn_dynamic_symbols_t syms, cudnnHandle_t handle);
  ~CuDNNModuleState();

  // Creates a new tensor for cuDNN graph argument.
  StatusOr<vm::ref<CuDNNTensor>> Argument(int64_t dtype,
                                          const vm::ref<iree_vm_list_t> dims,
                                          int64_t uid, int64_t alignment);

  // Creates a pointwise relu operation and returns result tensor.
  StatusOr<vm::ref<CuDNNTensor>> PointwiseRelu(const vm::ref<CuDNNTensor> input,
                                               float lower_clip,
                                               float upper_clip, int64_t uid,
                                               int64_t alignment);

  // TODO(ezhulenev): To be able to pass a list of tensors, `!cudnn.tensor` has
  // to be registered as a ref type (see `IREE::VM::RefType` and`!vmvx.buffer`
  // which is registered as reference type and can be added to the list).

  // Builds a cuDNN graph computing tensors.
  StatusOr<vm::ref<CuDNNOperationGraph>> BuildGraph(
      const vm::ref<CuDNNTensor> tensor);

  // Prints tensor debug information to stderr.
  Status PrintTensorDebug(const vm::ref<CuDNNTensor> tensor);

  // Prints graph debug information to stderr.
  Status PrintGraphDebug(const vm::ref<CuDNNOperationGraph> graph);

 private:
  CuDNNModuleState(const CuDNNModuleState&) = delete;
  CuDNNModuleState& operator=(const CuDNNModuleState&) = delete;

  openxla_cudnn_dynamic_symbols_t syms_;

  // IREE custom module state must be thread-compatible, and access to the same
  // state object will be synchronized by the caller, so we can safely access
  // cuDNN handle without any additional synchronization.
  cudnnHandle_t handle_;
};

CuDNNModuleState::CuDNNModuleState(openxla_cudnn_dynamic_symbols_t syms,
                                   cudnnHandle_t handle)
    : syms_(syms), handle_(handle) {}

CuDNNModuleState::~CuDNNModuleState() {
  CUDNN_STATUS_CHECK_OK(&syms_, cudnnDestroy(handle_));
}

static StatusOr<cudnnDataType_t> ToCudnnDataType(int64_t dtype) {
  if (dtype < CUDNN_DATA_FLOAT || dtype > CUDNN_DATA_FAST_FLOAT_FOR_FP8)
    return Status(StatusCode::kInvalidArgument, "unsupported data type");
  return static_cast<cudnnDataType_t>(dtype);
}

static StatusOr<std::vector<int64_t>> LoadI64Vec(const iree_vm_list_t* list) {
  std::vector<int64_t> vector(iree_vm_list_size(list));
  for (size_t i = 0; i < vector.size(); ++i) {
    iree_vm_value_t value;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_get_value_as(list, i, IREE_VM_VALUE_TYPE_I64, &value));
    vector[i] = value.i64;
  }
  return vector;
}

static std::vector<int64_t> GetRowMajorStrides(span<const int64_t> dims) {
  std::vector<int64_t> strides(dims.size(), 1);
  for (int64_t i = dims.size() - 2; i >= 0; --i)
    strides[i] = dims[i] * strides[i + 1];
  return strides;
}

StatusOr<vm::ref<CuDNNTensor>> CuDNNModuleState::Argument(
    int64_t dtype, const vm::ref<iree_vm_list_t> dims, int64_t uid,
    int64_t alignment) {
  IREE_ASSIGN_OR_RETURN(cudnnDataType_t data_type, ToCudnnDataType(dtype));
  IREE_ASSIGN_OR_RETURN(std::vector<int64_t> dimensions, LoadI64Vec(&*dims));
  std::vector<int64_t> strides = GetRowMajorStrides(dimensions);
  return CreateArgument(&syms_, dimensions, strides, uid, data_type, alignment);
}

Status CuDNNModuleState::PrintTensorDebug(const vm::ref<CuDNNTensor> tensor) {
  std::string desc = tensor->tensor().describe();
  fprintf(stderr, "Tensor: %s\n", desc.c_str());
  return OkStatus();
}

Status CuDNNModuleState::PrintGraphDebug(
    const vm::ref<CuDNNOperationGraph> graph) {
  std::string desc = graph->graph().describe();
  fprintf(stderr, "Graph: %s\n", desc.c_str());
  return OkStatus();
}

StatusOr<vm::ref<CuDNNTensor>> CuDNNModuleState::PointwiseRelu(
    const vm::ref<CuDNNTensor> input, float lower_clip, float upper_clip,
    int64_t uid, int64_t alignment) {
  return CreatePointwiseRelu(&syms_, *input, lower_clip, upper_clip, uid,
                             alignment);
}

StatusOr<vm::ref<CuDNNOperationGraph>> CuDNNModuleState::BuildGraph(
    const vm::ref<CuDNNTensor> tensor) {
  return CreateOperationGraph(&syms_, handle_, {tensor.get()});
}

static const vm::NativeFunction<CuDNNModuleState> kCuDNNModuleFunctions[] = {
    vm::MakeNativeFunction("tensor.arg", &CuDNNModuleState::Argument),
    vm::MakeNativeFunction("pointwise_relu", &CuDNNModuleState::PointwiseRelu),
    vm::MakeNativeFunction("build.graph", &CuDNNModuleState::BuildGraph),
    vm::MakeNativeFunction("debug.tensor", &CuDNNModuleState::PrintTensorDebug),
    vm::MakeNativeFunction("debug.graph", &CuDNNModuleState::PrintGraphDebug),
};

//===----------------------------------------------------------------------===//
// CuDNN module instance that will be allocated and reused across contexts.
//===----------------------------------------------------------------------===//

class CuDNNModule final : public vm::NativeModule<CuDNNModuleState> {
 public:
  CuDNNModule(iree_vm_instance_t* instance, iree_hal_device_t* device,
              iree_allocator_t host_allocator, CUcontext cuda_ctx);

  StatusOr<std::unique_ptr<CuDNNModuleState>> CreateState(
      iree_allocator_t host_allocator) override;

 private:
  static constexpr uint32_t kVersion = 0;

  using NativeModule = vm::NativeModule<CuDNNModuleState>;

  // Retain a reference to the HAL (CUDA) device to keep CUDA context wrapper
  // alive for the duration of cuDNN module lifetime.
  vm::ref<iree_hal_device_t> device_;

  // CUDA context bound to the instance of a HAL CUDA device.
  CUcontext cuda_ctx_;
};

CuDNNModule::CuDNNModule(iree_vm_instance_t* instance,
                         iree_hal_device_t* device,
                         iree_allocator_t host_allocator, CUcontext cuda_ctx)
    : NativeModule("cudnn", CuDNNModule::kVersion, instance, host_allocator,
                   {kCuDNNModuleFunctions}),
      device_(vm::retain_ref(device)),
      cuda_ctx_(cuda_ctx) {}

StatusOr<std::unique_ptr<CuDNNModuleState>> CuDNNModule::CreateState(
    iree_allocator_t host_allocator) {
  // Load cuDNN library and resolve API symbols.
  openxla_cudnn_dynamic_symbols_t syms;
  iree_status_t status =
      openxla_cudnn_dynamic_symbols_initialize(host_allocator, &syms);
  if (!iree_status_is_ok(status)) return status;

  // Create a cuDNN handle for the new state object.
  cudnnHandle_t handle;
  // TODO: We must guarantee that `cuda_ctx_` is current when we create cuDNN
  // handle. Currently we rely on implicit guarantee that module is loaded
  // immediately after device is created, however it might not always be true?
  CUDNN_RETURN_IF_ERROR(&syms, cudnnCreate(&handle), "cudnnCreate");

  return std::make_unique<CuDNNModuleState>(syms, handle);
}

}  // namespace openxla::runtime::nvgpu

//===----------------------------------------------------------------------===//
// Register cuDNN module with IREE runtime.
//===----------------------------------------------------------------------===//

using namespace openxla::runtime::nvgpu;

template <typename T>
static iree_status_t RegisterType(iree_vm_ref_type_descriptor_t* descriptor,
                                  const char* type_name) {
  if (descriptor->type == IREE_VM_REF_TYPE_NULL) {
    descriptor->type_name = iree_make_cstring_view(type_name);
    descriptor->offsetof_counter = T::offsetof_counter();
    descriptor->destroy = T::DirectDestroy;
    return iree_vm_ref_register_type(descriptor);
  }
  return iree_ok_status();
}

extern "C" iree_status_t iree_custom_module_cudnn_create(
    iree_vm_instance_t* instance, iree_hal_device_t* device,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);

  CUcontext cuda_ctx;
  IREE_RETURN_IF_ERROR(iree_hal_cuda_device_get_context(device, &cuda_ctx));
  auto module =
      std::make_unique<CuDNNModule>(instance, device, host_allocator, cuda_ctx);
  *out_module = module.release()->interface();

  return iree_ok_status();
}

extern "C" iree_status_t iree_custom_module_cudnn_register_types(
    iree_vm_instance_t* instance) {
  IREE_RETURN_IF_ERROR(
      RegisterType<CuDNNTensor>(&cudnn_tensor_descriptor, "cudnn.tensor"));
  IREE_RETURN_IF_ERROR(RegisterType<CuDNNOperationGraph>(
      &cudnn_operation_graph_descriptor, "cudnn.operation_graph"));
  return iree_ok_status();
}
