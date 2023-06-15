// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_CC_H_
#define OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_CC_H_

#include "iree/base/status_cc.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/ref_cc.h"
#include "openxla/runtime/async/async_runtime.h"
#include "tsl/concurrency/async_value.h"
#include "tsl/concurrency/async_value_ref.h"
#include "tsl/concurrency/chain.h"

namespace openxla::runtime::async {

// Returns the  IREE VM value type (eg, F32) corresponding to the given
// template parameter native type (eg, float).
template <typename NativeT>
iree_vm_value_type_t NativeToVMValueType() {
  // Make the expression depend on the template parameter NativeT so
  // that this compile-time error only appears if this function is
  // instantiated with some concrete type that is not specialized
  // below.
  static_assert(!std::is_same<NativeT, NativeT>::value,
                "Cannot map native type to vm type.");
  return IREE_VM_VALUE_TYPE_NONE;
}

template <>
inline iree_vm_value_type_t NativeToVMValueType<tsl::Chain>() {
  return IREE_VM_VALUE_TYPE_NONE;
}

template <>
inline iree_vm_value_type_t NativeToVMValueType<int8_t>() {
  return IREE_VM_VALUE_TYPE_I8;
}

template <>
inline iree_vm_value_type_t NativeToVMValueType<int16_t>() {
  return IREE_VM_VALUE_TYPE_I16;
}

template <>
inline iree_vm_value_type_t NativeToVMValueType<int32_t>() {
  return IREE_VM_VALUE_TYPE_I32;
}

template <>
inline iree_vm_value_type_t NativeToVMValueType<int64_t>() {
  return IREE_VM_VALUE_TYPE_I64;
}

template <>
inline iree_vm_value_type_t NativeToVMValueType<float>() {
  return IREE_VM_VALUE_TYPE_F32;
}

template <>
inline iree_vm_value_type_t NativeToVMValueType<double>() {
  return IREE_VM_VALUE_TYPE_F64;
}

// Convert async token to iree VM ref object
iree::StatusOr<iree::vm::ref<iree_async_value_t>> AsTokenValue(
    tsl::AsyncValueRef<tsl::Chain> value, iree_allocator_t host_allocator) {
  iree_async_value_t *val;
  tsl::AsyncValue *async_value = value.GetAsyncValue();
  iree_async_value_create(
      async_value, iree_vm_make_ref_type_def(iree_async_runtime_token_type()),
      host_allocator, &val);
  return iree::vm::ref<iree_async_value_t>(val);
}

// Convert async scalar value to iree VM ref object
template <typename T>
using EnableIfScalarType = typename std::enable_if_t<
    std::disjunction_v<std::is_same<T, float>, std::is_same<T, double>,
                       std::is_same<T, int8_t>, std::is_same<T, int16_t>,
                       std::is_same<T, int32_t>, std::is_same<T, int64_t>>>;

template <typename T, EnableIfScalarType<T> * = nullptr>
iree::StatusOr<iree::vm::ref<iree_async_value_t>> AsScalarValue(
    tsl::AsyncValueRef<T> value, iree_allocator_t host_allocator) {
  iree_async_value_t *val;
  tsl::AsyncValue *async_value = value.GetAsyncValue();
  iree_async_value_create(async_value,
                          iree_vm_make_value_type_def(NativeToVMValueType<T>()),
                          host_allocator, &val);
  return iree::vm::ref<iree_async_value_t>(val);
}

// Convert async value with custom type to iree VM ref object
iree::StatusOr<iree::vm::ref<iree_async_value_t>> AsRefValue(
    tsl::AsyncValueRef<iree_vm_ref_t> value, iree_allocator_t host_allocator) {
  iree_async_value_t *val;
  tsl::AsyncValue *async_value = value.GetAsyncValue();
  iree_async_value_create(async_value,
                          iree_vm_make_ref_type_def(IREE_VM_REF_TYPE_ANY),
                          host_allocator, &val);
  return iree::vm::ref<iree_async_value_t>(val);
}

}  // namespace openxla::runtime::async

#endif  // OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_CC_H_
