//===-- Passes.td - Conversion pass definition file --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDNN_CONVERSION_PASSES
#define CUDNN_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// CUDNNToLLLVM
//===----------------------------------------------------------------------===//

def ConvertCUDNNToLLVMPass : Pass<"convert-cudnn-to-llvm", "ModuleOp"> {
  let summary = "Convert CUDNN dialect to LLVM dialect";
  let description = [{
    This pass converts supported CUDNN ops to LLVM dialect instructions.
  }];
  let dependentDialects = ["LLVM::LLVMDialect"];
}

#endif // CUDNN_CONVERSION_PASSES
