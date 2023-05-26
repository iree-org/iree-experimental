// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/async/async_runtime_test.h"

#include <thread>

#include "openxla/runtime/async/async_runtime_cc.h"
#include "tfrt/concurrency/async_value_ref.h"

extern "C" iree_async_value_t *async_runtime_test() {
  tsl::AsyncValueRef<int32_t> value =
      tsl::MakeAvailableAsyncValueRef<int32_t>(42);
  return reinterpret_cast<iree_async_value_t *>(
      openxla::runtime::async::AsValue<int32_t>(value));
}