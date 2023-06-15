// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_ASYNC_TSL_ASYNC_VALUE_H_
#define OPENXLA_RUNTIME_ASYNC_TSL_ASYNC_VALUE_H_

#include "iree/base/api.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/vm/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef void *tsl_async_value_type_t;

IREE_API_EXPORT void wait_tsl_async_value(tsl_async_value_type_t value);

IREE_API_EXPORT void connect_tsl_async_value_with_iree_event(
    tsl_async_value_type_t value, iree_event_t *handle);

IREE_API_EXPORT void drop_ref_tsl_async_value(tsl_async_value_type_t value);
IREE_API_EXPORT void add_ref_tsl_async_value(tsl_async_value_type_t value);

IREE_API_EXPORT bool is_tsl_async_value_error(tsl_async_value_type_t value);

IREE_API_EXPORT bool is_tsl_async_value_available(tsl_async_value_type_t value);

IREE_API_EXPORT void set_tsl_async_value_available(
    tsl_async_value_type_t value);

IREE_API_EXPORT void set_tsl_async_value_error(tsl_async_value_type_t value);

IREE_API_EXPORT iree_status_t get_tsl_async_scalar_value(
    tsl_async_value_type_t value, iree_vm_value_type_t type, char *buffer);

IREE_API_EXPORT iree_status_t
get_tsl_async_ref_value(tsl_async_value_type_t value, iree_vm_ref_type_t type,
                        iree_vm_ref_t **out_ref);
#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // OPENXLA_RUNTIME_ASYNC_TSL_ASYNC_VALUE_H_
