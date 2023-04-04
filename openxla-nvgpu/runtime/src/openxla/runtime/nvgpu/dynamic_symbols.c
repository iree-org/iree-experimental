// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/nvgpu/dynamic_symbols.h"

#include <string.h>

#include "iree/base/internal/dynamic_library.h"
#include "iree/base/target_platform.h"
#include "iree/base/tracing.h"

static const char* kCuDNNLoaderSearchNames[] = {
#if defined(IREE_PLATFORM_WINDOWS)
    "cudnn.dll",
#else
    "libcudnn.so",
#endif  // IREE_PLATFORM_WINDOWS
};

#define concat(A, B) A B

static iree_status_t openxla_cudnn_dynamic_symbols_resolve_all(
    openxla_cudnn_dynamic_symbols_t* syms) {
#define CUDNN_PFN_DECL(cuDNNSymbolName, ...)                          \
  {                                                                   \
    static const char* kName = #cuDNNSymbolName;                      \
    IREE_RETURN_IF_ERROR(iree_dynamic_library_lookup_symbol(          \
        syms->cudnn_library, kName, (void**)&syms->cuDNNSymbolName)); \
  }

#include "openxla/runtime/nvgpu/dynamic_symbol_tables.h"  // IWYU pragma: export

#undef CUDNN_PFN_DECL
  return iree_ok_status();
}

iree_status_t openxla_cudnn_dynamic_symbols_initialize(
    iree_allocator_t host_allocator,
    openxla_cudnn_dynamic_symbols_t* out_syms) {
  IREE_TRACE_ZONE_BEGIN(z0);
  memset(out_syms, 0, sizeof(*out_syms));
  iree_status_t status = iree_dynamic_library_load_from_files(
      IREE_ARRAYSIZE(kCuDNNLoaderSearchNames), kCuDNNLoaderSearchNames,
      IREE_DYNAMIC_LIBRARY_FLAG_NONE, host_allocator, &out_syms->cudnn_library);
  if (iree_status_is_not_found(status)) {
    iree_status_ignore(status);
    IREE_TRACE_ZONE_END(z0);
    return iree_make_status(IREE_STATUS_UNAVAILABLE,
                            "cuDNN runtime library not available; ensure "
                            "installed and on path");
  }
  if (iree_status_is_ok(status)) {
    status = openxla_cudnn_dynamic_symbols_resolve_all(out_syms);
  }
  if (!iree_status_is_ok(status)) {
    openxla_cudnn_dynamic_symbols_deinitialize(out_syms);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

void openxla_cudnn_dynamic_symbols_deinitialize(
    openxla_cudnn_dynamic_symbols_t* syms) {
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_dynamic_library_release(syms->cudnn_library);
  memset(syms, 0, sizeof(*syms));
  IREE_TRACE_ZONE_END(z0);
}
