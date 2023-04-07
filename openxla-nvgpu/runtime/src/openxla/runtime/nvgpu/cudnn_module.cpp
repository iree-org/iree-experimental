// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/nvgpu/cudnn_module.h"

#include <iree/base/status.h>
#include <iree/base/status_cc.h>
#include <iree/vm/ref_cc.h>

#include <cstdio>

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

  Status Hello() {
    fprintf(stderr, "Hello from OpenXLA CuDNN Module!\n");
    return OkStatus();
  }

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

static const vm::NativeFunction<CuDNNModuleState> kCuDNNModuleFunctions[] = {
    vm::MakeNativeFunction("hello", &CuDNNModuleState::Hello),
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

extern "C" iree_status_t iree_custom_module_cudnn_create(
    iree_vm_instance_t* instance, iree_hal_device_t* device,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);

  CUcontext cuda_ctx;
  IREE_RETURN_IF_ERROR(iree_hal_cuda_device_get_context(device, &cuda_ctx));

  auto module = std::make_unique<openxla::runtime::nvgpu::CuDNNModule>(
      instance, device, host_allocator, cuda_ctx);
  *out_module = module.release()->interface();

  return iree_ok_status();
}
