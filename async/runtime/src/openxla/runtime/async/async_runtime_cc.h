// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_CC_H_
#define OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_CC_H_

#include "iree/vm/ref_cc.h"
#include "tfrt/concurrency/async_value.h"
#include "tfrt/concurrency/async_value_ref.h"
#include "tfrt/concurrency/chain.h"

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

template <typename T>
using EnableIfScalarType = typename std::enable_if_t<
    std::disjunction_v<std::is_same<T, float>, std::is_same<T, double>,
                       std::is_same<T, int8_t>, std::is_same<T, int16_t>,
                       std::is_same<T, int32_t>, std::is_same<T, int64_t>>>;

class AsyncValue : public iree::vm::RefObject<AsyncValue> {
 public:
  explicit AsyncValue(tsl::AsyncValueRef<tsl::Chain> &&u)
      : type_(IREE_VM_VALUE_TYPE_NONE), value_(u.ReleaseRCRef()) {}

  template <typename T, EnableIfScalarType<T> * = nullptr>
  explicit AsyncValue(tsl::AsyncValueRef<T> &&u)
      : type_(NativeToVMValueType<T>()), value_(u.ReleaseRCRef()) {}

  template <typename T>
  const T &get() {
    return value_->get<T>();
  }

  tsl::AsyncValue *GetAsyncValue() { return value_.get(); }

  bool IsError() const { return value_->IsError(); }

  iree_vm_value_type_t GetElementType() const { return type_; }

 private:
  iree_vm_value_type_t type_;
  tsl::RCReference<tsl::AsyncValue> value_;
};

template <typename T>
static AsyncValue *AsValue(tsl::AsyncValueRef<T> value) {
  return new AsyncValue(std::move(value));
}

}  // namespace openxla::runtime::async

#endif  // OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_CC_H_