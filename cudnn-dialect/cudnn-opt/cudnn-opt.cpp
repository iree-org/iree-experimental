//===- cudnn-opt.cpp --------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"

#include "CUDNN/IR/CUDNNDialect.h"

#define GEN_PASS_DECL_CONVERTCUDNNTOLLVMPASS
#include "CUDNN/Conversion/Passes.h.inc"

int main(int argc, char **argv) {
  mlir::registerConvertCUDNNToLLVMPass();
  mlir::registerAllPasses();

  mlir::DialectRegistry registry;
  registry.insert<mlir::cudnn::CUDNNDialect>();
  registerAllDialects(registry);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "CUDNN optimizer driver\n", registry));
}
