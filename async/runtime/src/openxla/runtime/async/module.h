// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_ASYNC_MODULE_H_
#define OPENXLA_RUNTIME_ASYNC_MODULE_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "iree/vm/api.h"
#include "openxla/runtime/async/async_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

IREE_API_EXPORT iree_status_t iree_async_runtime_module_create(
    iree_vm_instance_t* instance, iree_allocator_t host_allocator,
    iree_vm_module_t** out_module);
#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // OPENXLA_RUNTIME_ASYNC_MODULE_H_
