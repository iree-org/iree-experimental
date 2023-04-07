// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_RUNTIME_NVGPU_CUDNN_STUB_H_
#define OPENXLA_RUNTIME_NVGPU_CUDNN_STUB_H_

#include "openxla/runtime/nvgpu/dynamic_symbols.h"

namespace openxla::runtime::nvgpu {

// RAII helper to bind cuDNN stubs to dynamically resolved symbols.
class ScopedCuDNNStubs {
 public:
  ScopedCuDNNStubs(openxla_cudnn_dynamic_symbols_t* syms);
  ~ScopedCuDNNStubs();

  static openxla_cudnn_dynamic_symbols_t* syms();

 private:
  openxla_cudnn_dynamic_symbols_t* syms_;
};

}  // namespace openxla::runtime::nvgpu

#endif  // OPENXLA_RUNTIME_NVGPU_CUDNN_STUB_H_
