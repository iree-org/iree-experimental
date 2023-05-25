// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off

//Test function
EXPORT_FN("test.value", iree_async_runtime_module_test_async_value, v, r)

//Async function
EXPORT_FN("value.await.i32", iree_async_runtime_module_async_value_await_i32, r, i)
EXPORT_FN("value.await.token", iree_async_runtime_module_async_value_await_token, r, v)

EXPORT_FN("value.create.i32", iree_async_runtime_module_create_async_value_i32, v, r)
EXPORT_FN("value.create.token", iree_async_runtime_module_create_async_token, v, r)

EXPORT_FN("value.fail", iree_async_runtime_module_fail_async_value, r, v)
EXPORT_FN("value.query", iree_async_runtime_module_query_async_value, r, i)
EXPORT_FN("value.signal", iree_async_runtime_module_signal_async_value, r, v)

// clang-format on