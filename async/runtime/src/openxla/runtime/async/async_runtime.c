// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/async/async_runtime.h"

#include "openxla/runtime/async/tsl_async_value.h"

//===----------------------------------------------------------------------===//
// iree_async_value_t
//===----------------------------------------------------------------------===//

// Represent tsl::AsyncValue in iree VM to implement non-blocking wait on
// tsl::AsyncValue.
// When calling by async.await, it will setup the callback on
// tsl:AsyncValue and yield the execution
//
// When internal tsl::AsyncValue |value| becomes available, it will activate its
// iree_event_t handle |handle|. Scheduler can then schedule the continuation
// of the execution.
//
// |type| defines the internal type of tsl::AsyncValue
struct iree_async_value_t {
  iree_atomic_ref_count_t ref_count;
  iree_allocator_t host_allocator;
  iree_vm_type_def_t type;
  tsl_async_value_type_t value;
  iree_event_t handle;
};

struct iree_async_runtime_token_t {};

IREE_API_EXPORT iree_status_t iree_async_value_create(
    void *async_value, iree_vm_type_def_t type, iree_allocator_t host_allocator,
    iree_async_value_t **out_value) {
  IREE_ASSERT_ARGUMENT(out_value);
  *out_value = NULL;
  if (IREE_UNLIKELY(!async_value)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Empty tsl::AsyncValue");
  }

  IREE_TRACE_ZONE_BEGIN(z0);
  iree_async_value_t *value = NULL;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, iree_allocator_malloc(host_allocator, sizeof(iree_async_value_t),
                                (void **)&value));
  iree_atomic_ref_count_init(&value->ref_count);
  value->host_allocator = host_allocator;
  value->type = type;
  value->value = async_value;
  add_ref_tsl_async_value(value->value);
  // TODO: We may want to use iree_event_pool in the future
  iree_event_initialize(false, &value->handle);
  connect_tsl_async_value_with_iree_event(async_value, &value->handle);
  *out_value = value;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_async_value_destroy(iree_async_value_t *value) {
  IREE_ASSERT_ARGUMENT(value);
  IREE_ASSERT_REF_COUNT_ZERO(&value->ref_count);
  IREE_TRACE_ZONE_BEGIN(z0);

  drop_ref_tsl_async_value(value->value);
  iree_event_deinitialize(&value->handle);
  iree_allocator_t host_allocator = value->host_allocator;
  iree_allocator_free(host_allocator, value);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT void iree_async_value_retain(iree_async_value_t *value) {
  IREE_ASSERT_ARGUMENT(value);
  if (IREE_LIKELY(value)) {
    iree_atomic_ref_count_inc(&value->ref_count);
  }
}

IREE_API_EXPORT void iree_async_value_release(iree_async_value_t *value) {
  if (IREE_LIKELY(value) && iree_atomic_ref_count_dec(&value->ref_count) == 1) {
    iree_async_value_destroy(value);
  }
}

IREE_API_EXPORT iree_status_t
iree_async_value_query(iree_async_value_t *value) {
  if (!value) return iree_ok_status();

  if (is_tsl_async_value_error(value->value)) {
    return iree_status_from_code(IREE_STATUS_INTERNAL);
  } else if (!is_tsl_async_value_available(value->value)) {
    return iree_status_from_code(IREE_STATUS_DEFERRED);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_async_value_signal(iree_async_value_t *value) {
  IREE_ASSERT_ARGUMENT(value);
  set_tsl_async_value_available(value->value);
  return iree_ok_status();
}

IREE_API_EXPORT void iree_async_value_fail(iree_async_value_t *value) {
  set_tsl_async_value_error(value->value);
}

IREE_API_EXPORT iree_status_t iree_async_value_wait(iree_async_value_t *value,
                                                    iree_timeout_t timeout) {
  IREE_ASSERT_ARGUMENT(value);
  IREE_TRACE_ZONE_BEGIN(z0);
  wait_tsl_async_value(value->value);
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_async_value_get_scalar_value(
    iree_async_value_t *value, iree_vm_value_type_t type, char *buffer) {
  IREE_ASSERT_ARGUMENT(value);
  if (!iree_vm_type_def_equal(value->type, iree_vm_make_value_type_def(type))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Mismatched async value type");
  }
  return get_tsl_async_scalar_value(value->value, type, buffer);
}

IREE_API_EXPORT iree_status_t iree_async_value_get_ref_value(
    iree_async_value_t *value, iree_vm_ref_type_t type,
    iree_vm_ref_t **out_ref) {
  IREE_ASSERT_ARGUMENT(value);
  if (!iree_vm_type_def_equal(value->type, iree_vm_make_ref_type_def(type))) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "Mismatched async value type");
  }

  return get_tsl_async_ref_value(value->value, type, out_ref);
}

IREE_API_EXPORT iree_status_t iree_async_value_export(
    iree_async_value_t *value, iree_wait_primitive_t *out_wait_primitive) {
  out_wait_primitive->type = value->handle.type;
  out_wait_primitive->value = value->handle.value;
  return iree_ok_status();
}

iree_status_t iree_async_value_wait_source_ctl(
    iree_wait_source_t wait_source, iree_wait_source_command_t command,
    const void *params, void **inout_ptr) {
  iree_async_value_t *value = (iree_async_value_t *)wait_source.self;
  switch (command) {
    case IREE_WAIT_SOURCE_COMMAND_QUERY: {
      iree_status_code_t *out_wait_status_code =
          (iree_status_code_t *)inout_ptr;
      iree_status_t status = iree_async_value_query(value);
      if (!iree_status_is_ok(status)) {
        *out_wait_status_code = iree_status_code(status);
        iree_status_ignore(status);
      } else {
        *out_wait_status_code = IREE_STATUS_OK;
      }
      return iree_ok_status();
    }
    case IREE_WAIT_SOURCE_COMMAND_WAIT_ONE: {
      const iree_timeout_t timeout =
          ((const iree_wait_source_wait_params_t *)params)->timeout;
      return iree_async_value_wait(value, timeout);
    }
    case IREE_WAIT_SOURCE_COMMAND_EXPORT: {
      iree_wait_primitive_t *out_wait_primitive =
          (iree_wait_primitive_t *)inout_ptr;
      return iree_async_value_export(value, out_wait_primitive);
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "Unimplemented wait_source command");
  }
}

IREE_API_EXPORT iree_wait_source_t
iree_async_value_await(iree_async_value_t *value) {
  if (!value) return iree_wait_source_immediate();
  iree_wait_source_t wait_source;
  wait_source.self = value;
  wait_source.data = 0;
  wait_source.ctl = iree_async_value_wait_source_ctl;
  return wait_source;
}

IREE_VM_DEFINE_TYPE_ADAPTERS(iree_async_value, iree_async_value_t);
IREE_VM_DEFINE_TYPE_ADAPTERS(iree_async_runtime_token,
                             iree_async_runtime_token_t);

static iree_status_t register_async_value_type(
    iree_vm_instance_t *instance, const char *type_name,
    iree_vm_ref_type_t *out_registration) {
  static iree_vm_ref_type_descriptor_t descriptor = {0};

  descriptor.type_name = iree_make_cstring_view(type_name);
  descriptor.offsetof_counter =
      offsetof(iree_async_value_t, ref_count) / IREE_VM_REF_COUNTER_ALIGNMENT;
  descriptor.destroy = (iree_vm_ref_destroy_t)iree_async_value_destroy;

  return iree_vm_instance_register_type(instance, &descriptor,
                                        out_registration);
}

IREE_API_EXPORT iree_status_t
openxla_async_runtime_module_register_types(iree_vm_instance_t *instance) {
  IREE_RETURN_IF_ERROR(register_async_value_type(
      instance, "async.value", &iree_async_value_registration));
  return iree_ok_status();
}
