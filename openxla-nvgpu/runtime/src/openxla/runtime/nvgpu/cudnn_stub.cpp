// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/nvgpu/cudnn_stub.h"

#include "openxla/runtime/nvgpu/dynamic_symbols.h"

namespace openxla::runtime::nvgpu {

static thread_local openxla_cudnn_dynamic_symbols_t* cudnn_syms = nullptr;

ScopedCuDNNStubs::ScopedCuDNNStubs(openxla_cudnn_dynamic_symbols_t* syms)
    : syms_(cudnn_syms) {
  cudnn_syms = syms;
}

ScopedCuDNNStubs::~ScopedCuDNNStubs() { cudnn_syms = syms_; }

openxla_cudnn_dynamic_symbols_t* ScopedCuDNNStubs::syms() { return cudnn_syms; }

}  // namespace openxla::runtime::nvgpu
