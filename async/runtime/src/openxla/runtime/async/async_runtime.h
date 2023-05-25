// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_H_
#define OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_H_

#include "iree/base/api.h"
#include "iree/base/status.h"
#include "iree/vm/api.h"
#include "iree/vm/ref.h"
#include "iree/vm/value.h"

typedef struct iree_async_value_t iree_async_value_t;
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_async_value, iree_async_value_t);

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_async_value_t api
//===----------------------------------------------------------------------===//

IREE_API_EXPORT iree_status_t
iree_async_value_create_i32(iree_async_value_t **out_value);

IREE_API_EXPORT iree_status_t
iree_async_value_create_token(iree_async_value_t **out_token);

IREE_API_EXPORT iree_status_t iree_async_value_get_scalar_value(
    iree_async_value_t *value, iree_vm_value_type_t type, char *buffer);

IREE_API_EXPORT iree_status_t iree_async_value_query(iree_async_value_t *value);

IREE_API_EXPORT iree_status_t
iree_async_value_signal(iree_async_value_t *value);

IREE_API_EXPORT void iree_async_value_fail(iree_async_value_t *value);

IREE_API_EXPORT iree_status_t iree_async_value_wait(iree_async_value_t *value,
                                                    iree_timeout_t timeout);

// Releases |token| and destroys it if the caller is the last owner.
IREE_API_EXPORT void iree_async_value_release(iree_async_value_t *value);

IREE_API_EXPORT iree_status_t iree_async_value_and_then(
    iree_async_value_t *value, iree_loop_callback_t callback, iree_loop_t loop);

// Returns a wait source reference to |async_value|
// The async_value must be kept live for as long as the reference is live
IREE_API_EXPORT iree_wait_source_t
iree_async_value_await(iree_async_value_t *value);

IREE_API_EXPORT iree_status_t iree_async_value_wait_source_ctl(
    iree_wait_source_t wait_source, iree_wait_source_command_t command,
    const void *params, void **inout_ptr);

//===----------------------------------------------------------------------===//
// iree_async_value_t implementation details
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_async_value_destroy(iree_async_value_t *value);
IREE_API_EXPORT uint32_t iree_async_value_offsetof_counter();

// Registers the custom types used by the full async module.
// WARNING: not thread-safe; call at startup before using.
IREE_API_EXPORT iree_status_t
openxla_async_runtime_module_register_types(iree_vm_instance_t *instance);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_H_