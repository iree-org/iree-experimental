// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_CPU_CLIENT_H_
#define IREE_PJRT_PLUGIN_PJRT_CPU_CLIENT_H_

#include "iree/hal/drivers/local_task/task_driver.h"
#include "iree/integrations/pjrt/common/api_impl.h"

namespace iree::pjrt::cpu {

class CPUClientInstance final : public ClientInstance {
 public:
  CPUClientInstance(Globals& globals);
  ~CPUClientInstance();
  PJRT_Error* CreateDriver(iree_hal_driver_t** out_driver) override;

 private:
  iree_status_t InitializeDeps();

  // Options.
  iree_hal_task_device_params_t device_params_;
  iree_task_executor_options_t task_executor_options_;
  iree_task_topology_t task_topology_options_;

  // Deps.
  iree_allocator_t host_allocator_;
  iree_hal_executable_loader_t* loaders_[8] = {nullptr};
  iree_host_size_t loader_count_ = 0;
  iree_task_executor_t* executor_ = nullptr;
  iree_hal_allocator_t* device_allocator_ = nullptr;
};

}  // namespace iree::pjrt::cpu

#endif  // IREE_PJRT_PLUGIN_PJRT_CPU_CLIENT_H_
