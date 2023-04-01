// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include "mlir/Pass/PassBase.td"

#ifndef OPENXLA_NVGPU_TRANSFORMS_PASSES_TD_
#define OPENXLA_NVGPU_TRANSFORMS_PASSES_TD_

def Sample : Pass<"openxla-nvgpu-sample", "mlir::ModuleOp"> {
  let summary = "Sample pass";
  let constructor = [{
    ::openxla::compiler::nvgpu::createSamplePass()
  }];
}

#endif // OPENXLA_NVGPU_TRANSFORMS_PASSES_TD_
