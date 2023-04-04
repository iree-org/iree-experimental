// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/nvgpu/status_util.h"

#include <stddef.h>

#include "openxla/runtime/nvgpu/dynamic_symbols.h"

iree_status_t openxla_cudnn_status_to_status(
    openxla_cudnn_dynamic_symbols_t* syms, cudnnStatus_t status,
    const char* file, uint32_t line) {
  if (IREE_LIKELY(status == CUDNN_STATUS_SUCCESS)) {
    return iree_ok_status();
  }

  const char* error_string = syms->cudnnGetErrorString(status);
  return iree_make_status_with_location(file, line, IREE_STATUS_INTERNAL,
                                        "cuDNN error '%s' (%d)", error_string,
                                        status);
}
