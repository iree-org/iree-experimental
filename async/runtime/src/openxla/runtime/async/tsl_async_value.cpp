// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/async/tsl_async_value.h"

#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "tsl/concurrency/async_value.h"
#include "tsl/concurrency/async_value_ref.h"
#include "tsl/concurrency/chain.h"

using namespace tsl;

IREE_API_EXPORT void wait_tsl_async_value(tsl_async_value_type_t value) {
  AsyncValue *async_val = static_cast<AsyncValue *>(value);
  tsl::BlockUntilReady(async_val);
}

IREE_API_EXPORT void connect_tsl_async_value_with_iree_event(
    tsl_async_value_type_t value, iree_event_t *handle) {
  AsyncValue *async_val = static_cast<AsyncValue *>(value);
  async_val->AndThen([&]() { iree_event_set(handle); });
}

IREE_API_EXPORT void drop_ref_tsl_async_value(tsl_async_value_type_t value) {
  AsyncValue *async_val = static_cast<AsyncValue *>(value);
  async_val->DropRef();
}

IREE_API_EXPORT void add_ref_tsl_async_value(tsl_async_value_type_t value) {
  AsyncValue *async_val = static_cast<AsyncValue *>(value);
  async_val->AddRef();
}

IREE_API_EXPORT bool is_tsl_async_value_error(tsl_async_value_type_t value) {
  AsyncValue *async_val = static_cast<AsyncValue *>(value);
  return async_val->IsError();
}

IREE_API_EXPORT bool is_tsl_async_value_available(
    tsl_async_value_type_t value) {
  AsyncValue *async_val = static_cast<AsyncValue *>(value);
  return async_val->IsAvailable();
}

IREE_API_EXPORT void set_tsl_async_value_available(
    tsl_async_value_type_t value) {
  AsyncValue *async_val = static_cast<AsyncValue *>(value);
  async_val->SetStateConcrete();
}

IREE_API_EXPORT void set_tsl_async_value_error(tsl_async_value_type_t value) {
  AsyncValue *async_val = static_cast<AsyncValue *>(value);
  async_val->SetError(absl::InternalError("async runtime error"));
}

IREE_API_EXPORT iree_status_t get_tsl_async_scalar_value(
    tsl_async_value_type_t value, iree_vm_value_type_t type, char *buffer) {
  AsyncValue *async_val = static_cast<AsyncValue *>(value);
  switch (type) {
    case IREE_VM_VALUE_TYPE_NONE:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "No value to extract for VM value type NONE");
      break;
    case IREE_VM_VALUE_TYPE_I8:
      *buffer = async_val->get<int8_t>();
      break;
    case IREE_VM_VALUE_TYPE_I16:
      *buffer = async_val->get<int16_t>();
      break;
    case IREE_VM_VALUE_TYPE_I32:
      *buffer = async_val->get<int32_t>();
      break;
    case IREE_VM_VALUE_TYPE_I64:
      *buffer = async_val->get<int64_t>();
      break;
    case IREE_VM_VALUE_TYPE_F32:
      *buffer = async_val->get<float>();
      break;
    case IREE_VM_VALUE_TYPE_F64:
      *buffer = async_val->get<double>();
      break;
    default:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                              "Unsupported scalar type for async value");
  }

  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t
get_tsl_async_ref_value(tsl_async_value_type_t value, iree_vm_ref_type_t type,
                        iree_vm_ref_t **out_ref) {
  AsyncValue *async_val = static_cast<AsyncValue *>(value);
  *out_ref = &async_val->get<iree_vm_ref_t>();
  return iree_ok_status();
}
