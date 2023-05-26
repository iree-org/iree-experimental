// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/async/async_runtime.h"

#include <cstddef>
#include <optional>

#include "openxla/runtime/async/async_runtime_cc.h"
#include "tfrt/concurrency/async_value.h"
#include "tfrt/concurrency/async_value_ref.h"
#include "tfrt/concurrency/chain.h"

using openxla::runtime::async::AsyncValue;

IREE_API_EXPORT iree_status_t
iree_async_value_create_token(iree_async_value_t **out_token) {
  tsl::AsyncValueRef<tsl::Chain> chain =
      tsl::MakeConstructedAsyncValueRef<tsl::Chain>();
  AsyncValue *val = new AsyncValue(std::move(chain));
  *out_token = reinterpret_cast<iree_async_value_t *>(val);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_async_value_create_i32(iree_async_value_t **out_value) {
  tsl::AsyncValueRef<int32_t> async_ref =
      tsl::MakeConstructedAsyncValueRef<int32_t>();
  AsyncValue *val = new AsyncValue(std::move(async_ref));
  *out_value = reinterpret_cast<iree_async_value_t *>(val);
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_async_value_get_scalar_value(
    iree_async_value_t *value, iree_vm_value_type_t type, char *buffer) {
  AsyncValue *val = reinterpret_cast<AsyncValue *>(value);
  IREE_ASSERT_ARGUMENT(val);
  if (val->GetElementType() != type) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "Failed to extract async value due to incompatible type");
  }
  *buffer = val->get<int32_t>();
  return iree_ok_status();
}

IREE_API_EXPORT void iree_async_value_destroy(iree_async_value_t *value) {
  AsyncValue *val = reinterpret_cast<AsyncValue *>(value);
  IREE_ASSERT_ARGUMENT(val);
  delete val;
}

IREE_API_EXPORT uint32_t iree_async_value_offsetof_counter() {
  return AsyncValue::offsetof_counter();
}

IREE_API_EXPORT void iree_async_value_release(iree_async_value_t *value) {
  AsyncValue *val = reinterpret_cast<AsyncValue *>(value);
  val->ReleaseReference();
}

IREE_API_EXPORT iree_status_t
iree_async_value_query(iree_async_value_t *value) {
  AsyncValue *val = reinterpret_cast<AsyncValue *>(value);
  if (!val) return iree_ok_status();
  if (!val->GetAsyncValue()->IsAvailable()) {
    return iree_status_from_code(IREE_STATUS_DEFERRED);
  }
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_async_value_signal(iree_async_value_t *token) {
  AsyncValue *val = reinterpret_cast<AsyncValue *>(token);
  val->GetAsyncValue()->SetStateConcrete();
  return iree_ok_status();
}

IREE_API_EXPORT void iree_async_value_fail(iree_async_value_t *value) {
  AsyncValue *val = reinterpret_cast<AsyncValue *>(value);
  val->GetAsyncValue()->SetError(absl::InternalError("async runtime error"));
}

IREE_API_EXPORT iree_status_t iree_async_value_wait(iree_async_value_t *value,
                                                    iree_timeout_t timeout) {
  assert(false && "unimplemented");
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
iree_async_value_and_then(iree_async_value_t *value,
                          iree_loop_callback_t callback, iree_loop_t loop) {
  AsyncValue *val = reinterpret_cast<AsyncValue *>(value);
  val->GetAsyncValue()->AndThen([callback, loop]() {
    iree_status_t status =
        callback.fn(callback.user_data, loop, iree_ok_status());
    (void)status;
    // notify loop is status is not OK
  });
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_async_value_wait_source_ctl(
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
      const iree_wait_primitive_type_t target_type =
          ((const iree_wait_source_export_params_t *)params)->target_type;
      // TODO(benvanik): support exporting fences to real wait handles.
      iree_wait_primitive_t *out_wait_primitive =
          (iree_wait_primitive_t *)inout_ptr;
      memset(out_wait_primitive, 0, sizeof(*out_wait_primitive));
      (void)target_type;
      return iree_make_status(IREE_STATUS_UNAVAILABLE,
                              "requested wait primitive type %d is unavailable",
                              (int)target_type);
    }
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "unimplemented wait_source command");
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

static iree_status_t RegisterAsyncValueType(
    iree_vm_instance_t *instance, const char *type_name,
    iree_vm_ref_type_t *out_registration) {
  static iree_vm_ref_type_descriptor_t descriptor = {0};

  descriptor.type_name = iree_make_cstring_view(type_name);
  descriptor.offsetof_counter = AsyncValue::offsetof_counter();
  descriptor.destroy = AsyncValue::DirectDestroy;

  return iree_vm_instance_register_type(instance, &descriptor,
                                        out_registration);
}

extern "C" iree_status_t openxla_async_runtime_module_register_types(
    iree_vm_instance_t *instance) {
  IREE_RETURN_IF_ERROR(RegisterAsyncValueType(instance, "async.value",
                                              &iree_async_value_registration));
  return iree_ok_status();
}