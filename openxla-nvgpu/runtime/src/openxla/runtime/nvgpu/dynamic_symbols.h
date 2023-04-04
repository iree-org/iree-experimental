// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_NVGPU_DYNAMIC_SYMBOLS_H_
#define OPENXLA_RUNTIME_NVGPU_DYNAMIC_SYMBOLS_H_

#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"
#include "openxla/runtime/nvgpu/cudnn_headers.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// `openxla_cudnn_dynamic_symbols_t` allow loading dynamically a subset of cuDNN
// API. It loads all the function declared in `dynamic_symbol_tables.h` and fail
// if any of the symbol is not available. The functions signatures are matching
// the declarations in `cudnn.h`. This mechanism is based on dynamic CUDA
// symbols loading in the IREE HAL backend.
typedef struct openxla_cudnn_dynamic_symbols_t {
  iree_dynamic_library_t* cudnn_library;

#define CUDNN_PFN_DECL(cuDNNSymbolName, ...) \
  cudnnStatus_t (*cuDNNSymbolName)(__VA_ARGS__);
#define CUDNN_PFN_DECL_STR_RETURN(cuDNNSymbolName, ...) \
  const char* (*cuDNNSymbolName)(__VA_ARGS__);

#include "openxla/runtime/nvgpu/dynamic_symbol_tables.h"  // IWYU pragma: export

#undef CUDNN_PFN_DECL
} openxla_cudnn_dynamic_symbols_t;

// Initializes |out_syms| in-place with dynamically loaded cuDNN symbols.
// openxla_cudnn_dynamic_symbols_deinitialize must be used to release the
// library resources.
iree_status_t openxla_cudnn_dynamic_symbols_initialize(
    iree_allocator_t host_allocator, openxla_cudnn_dynamic_symbols_t* out_syms);

// Deinitializes |syms| by unloading the backing library. All function pointers
// will be invalidated. They _may_ still work if there are other reasons the
// library remains loaded so be careful.
void openxla_cudnn_dynamic_symbols_deinitialize(
    openxla_cudnn_dynamic_symbols_t* syms);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // OPENXLA_RUNTIME_NVGPU_DYNAMIC_SYMBOLS_H_
