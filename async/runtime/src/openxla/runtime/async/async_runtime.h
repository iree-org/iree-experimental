// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_H_
#define OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_H_

#include "iree/base/api.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/base/status.h"
#include "iree/vm/api.h"
#include "iree/vm/ref.h"
#include "iree/vm/value.h"

typedef struct iree_async_value_t iree_async_value_t;
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_async_value, iree_async_value_t);

typedef struct iree_async_runtime_token_t iree_async_runtime_token_t;
IREE_VM_DECLARE_TYPE_ADAPTERS(iree_async_runtime_token,
                              iree_async_runtime_token_t);

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// iree_async_value_t api
//===----------------------------------------------------------------------===//

// Create iree_asycn_value_t with tsl::AsyncValue |async_value| and iree VM
// type |type|.
IREE_API_EXPORT iree_status_t iree_async_value_create(
    void *async_value, iree_vm_type_def_t type, iree_allocator_t host_allocator,
    iree_async_value_t **out_value);

// Retains the |value| for the caller.
IREE_API_EXPORT void iree_async_value_retain(iree_async_value_t *value);

// Releases |value| and destroys it if the caller is the last owner.
IREE_API_EXPORT void iree_async_value_release(iree_async_value_t *value);

// Query the status of the |value|
// Returns OK if internal tsl::AsyncValue is available, IREE_STATUS_DEFERRED if
// it is unavailable, or a failure if the internal value is in error state.
IREE_API_EXPORT iree_status_t iree_async_value_query(iree_async_value_t *value);

// Signal that |value| becomes available.
IREE_API_EXPORT iree_status_t
iree_async_value_signal(iree_async_value_t *value);

// Signal a |value| to indicate it has some error.
IREE_API_EXPORT void iree_async_value_fail(iree_async_value_t *value);

// Blocking wait on |value| to become available or return if timeout is reached
IREE_API_EXPORT iree_status_t iree_async_value_wait(iree_async_value_t *value,
                                                    iree_timeout_t timeout);

// Get the scalar value stored inside |value|. The |value| must be in available
// state.
IREE_API_EXPORT iree_status_t iree_async_value_get_scalar_value(
    iree_async_value_t *value, iree_vm_value_type_t type, char *buffer);

// Get the ref stored inside |value|. The |value| must be in available state.
IREE_API_EXPORT iree_status_t iree_async_value_get_ref_value(
    iree_async_value_t *value, iree_vm_ref_type_t type,
    iree_vm_ref_t **out_ref);

// Export |value| as wait_handle (iree_event_t in our case).
IREE_API_EXPORT iree_status_t iree_async_value_export(
    iree_async_value_t *value, iree_wait_primitive_t *out_wait_primitive);

IREE_API_EXPORT iree_status_t iree_async_value_wait_source_ctl(
    iree_wait_source_t wait_source, iree_wait_source_command_t command,
    const void *params, void **inout_ptr);

// Returns a wait source reference to |value|
// The value must be kept live for as long as the reference is live
IREE_API_EXPORT iree_wait_source_t
iree_async_value_await(iree_async_value_t *value);

// Export async value as system wait handle
IREE_API_EXPORT iree_status_t iree_async_value_export(
    iree_async_value_t *value, iree_wait_primitive_t *out_wait_primitive);

//===----------------------------------------------------------------------===//
// iree_async_value_t implementation details
//===----------------------------------------------------------------------===//

IREE_API_EXPORT void iree_async_value_destroy(iree_async_value_t *value);

// Registers the custom types used by the full async module.
// WARNING: not thread-safe; call at startup before using.
IREE_API_EXPORT iree_status_t
openxla_async_runtime_module_register_types(iree_vm_instance_t *instance);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_H_
