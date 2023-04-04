// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_NVGPU_STATUS_UTIL_H_
#define OPENXLA_RUNTIME_NVGPU_STATUS_UTIL_H_

#include <stdint.h>

#include "iree/base/api.h"
#include "openxla/runtime/nvgpu/dynamic_symbols.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Converts a cudnnStatus_t to an iree_status_t.
//
// Usage:
//   iree_status_t status = CUDNN_STATUS_TO_STATUS(cuDnnDoThing(...));
#define CUDNN_STATUS_TO_STATUS(syms, expr, ...) \
  openxla_cudnn_status_to_status((syms), ((syms)->expr), __FILE__, __LINE__)

// IREE_RETURN_IF_ERROR but implicitly converts the cudnnStatus_t return value
// to a iree_status_t.
//
// Usage:
//   CUDNN_RETURN_IF_ERROR(cuDnnDoThing(...));
#define CUDNN_RETURN_IF_ERROR(syms, expr, ...)                                \
  IREE_RETURN_IF_ERROR(openxla_cudnn_status_to_status((syms), ((syms)->expr), \
                                                      __FILE__, __LINE__),    \
                       __VA_ARGS__)

// IREE_CHECK_OK but implicitly converts the cudnnStatus_t return value to a
// iree_status_t.
//
// Usage:
//   CUDNN_STATUS_CHECK_OK(cuDnnDoThing(...));
#define CUDNN_STATUS_CHECK_OK(syms, expr, ...)                         \
  IREE_CHECK_OK(openxla_cudnn_status_to_status((syms), ((syms)->expr), \
                                               __FILE__, __LINE__))

// Converts a cudnnStatus_t to an iree_status_t.
iree_status_t openxla_cudnn_status_to_status(
    openxla_cudnn_dynamic_symbols_t* syms, cudnnStatus_t status,
    const char* file, uint32_t line);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // OPENXLA_RUNTIME_NVGPU_STATUS_UTIL_H_
