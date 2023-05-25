// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_TEST_H_
#define OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_TEST_H_

#include "openxla/runtime/async/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

iree_async_value_t *async_runtime_test();

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // OPENXLA_RUNTIME_ASYNC_ASYNCRUNTIME_TEST_H_