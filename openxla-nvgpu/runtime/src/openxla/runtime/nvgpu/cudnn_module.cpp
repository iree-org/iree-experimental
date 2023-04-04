// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/nvgpu/cudnn_module.h"

#include <iree/base/status_cc.h>

#include <cstdio>

#include "iree/vm/native_module_cc.h"

namespace openxla::runtime::nvgpu {

using namespace iree;

//===----------------------------------------------------------------------===//
// CuDNN module state encapsulates all the state required for running cuDNN
// operations (launching cuDNN graphs on a stream) at run time.
//===----------------------------------------------------------------------===//

class CuDNNModuleState {
 public:
  Status Hello() {
    fprintf(stderr, "Hello from OpenXLA CuDNN Module!\n");
    return OkStatus();
  }
};

static const vm::NativeFunction<CuDNNModuleState> kCuDNNModuleFunctions[] = {
    vm::MakeNativeFunction("hello", &CuDNNModuleState::Hello),
};

//===----------------------------------------------------------------------===//
// CuDNN module instance that will be allocated and reused across contexts.
//===----------------------------------------------------------------------===//

class CuDNNModule final : public vm::NativeModule<CuDNNModuleState> {
 public:
  using vm::NativeModule<CuDNNModuleState>::NativeModule;

  StatusOr<std::unique_ptr<CuDNNModuleState>> CreateState(
      iree_allocator_t host_allocator) override {
    return std::make_unique<CuDNNModuleState>();
  }
};

}  // namespace openxla::runtime::nvgpu

//===----------------------------------------------------------------------===//
// Register cuDNN module with IREE runtime.
//===----------------------------------------------------------------------===//

using namespace openxla::runtime::nvgpu;

extern "C" iree_status_t iree_custom_module_cudnn_create(
    iree_vm_instance_t* instance, iree_hal_device_t* device,
    iree_allocator_t host_allocator, iree_vm_module_t** out_module) {
  IREE_ASSERT_ARGUMENT(out_module);
  auto module = std::make_unique<CuDNNModule>(
      "cudnn", /*version=*/0, instance, host_allocator,
      span<const vm::NativeFunction<CuDNNModuleState>>(kCuDNNModuleFunctions));
  *out_module = module.release()->interface();
  return iree_ok_status();
}
