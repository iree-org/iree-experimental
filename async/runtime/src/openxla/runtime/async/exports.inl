// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// clang-format off

//Async function
EXPORT_FN("value.await", iree_async_runtime_module_async_value_await, r, v)

EXPORT_FN("value.fail", iree_async_runtime_module_fail_async_value, r, v)

EXPORT_FN("value.load.i32", iree_async_runtime_module_load_async_value_i32, r, i)
EXPORT_FN("value.load.ref", iree_async_runtime_module_load_async_value_ref, r, r)

EXPORT_FN("value.query", iree_async_runtime_module_query_async_value, r, i)
EXPORT_FN("value.signal", iree_async_runtime_module_signal_async_value, r, v)

// clang-format on
