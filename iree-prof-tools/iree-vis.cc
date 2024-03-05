// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <string>

#include "compiler/bindings/c/iree/compiler/embedding_api.h"
#include "compiler/bindings/c/iree/compiler/mlir_interop.h"
#include "third_party/abseil-cpp/absl/base/log_severity.h"
#include "third_party/abseil-cpp/absl/log/globals.h"
#include "third_party/abseil-cpp/absl/log/initialize.h"
#include "third_party/abseil-cpp/absl/log/log.h"
#include "third_party/abseil-cpp/absl/flags/flag.h"
#include "third_party/abseil-cpp/absl/flags/parse.h"
#include "third_party/abseil-cpp/absl/strings/str_cat.h"
#include "third_party/llvm-project/mlir/include/mlir/IR/Operation.h"

ABSL_FLAG(std::string, input_iree_file, "",
          "IREE MLIR text assembly file to read.");
ABSL_FLAG(std::string, iree_phase, "executable-sources",
          "IREE compilation phase of input IREE MLIR file.");

void Walk(mlir::Operation* op, absl::string_view indent) {
  LOG(INFO) << indent << op->getName().getStringRef().str();
  std::string child_indent = absl::StrCat(indent, "  ");
  op->walk([op, indent = absl::string_view(child_indent)](
      mlir::Operation* nested_op) {
    if (nested_op != op) {
      Walk(nested_op, indent);
    }
  });
}

int main(int argc, char** argv) {
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();
  absl::ParseCommandLine(argc, argv);

  ireeCompilerGlobalInitialize();
  auto* session = ireeCompilerSessionCreate();

  std::string input_iree_file = absl::GetFlag(FLAGS_input_iree_file);
  iree_compiler_source_t* source;
  auto* error =
      ireeCompilerSourceOpenFile(session, input_iree_file.data(), &source);
  if (error != nullptr) {
    LOG(ERROR) << "Can't load input file, " << input_iree_file
               << ": " << ireeCompilerErrorGetMessage(error);
    ireeCompilerErrorDestroy(error);
    ireeCompilerSessionDestroy(session);
    ireeCompilerGlobalShutdown();
    return 1;
  }

  auto* invoke = ireeCompilerInvocationCreate(session);
  ireeCompilerInvocationSetCompileFromPhase(
      invoke, absl::GetFlag(FLAGS_iree_phase).data());
  if (ireeCompilerInvocationParseSource(invoke, source)) {
    LOG(INFO) << "Parsing " << input_iree_file << " done successfully.";
    auto op = ireeCompilerInvocationExportStealModule(invoke);
    Walk(reinterpret_cast<mlir::Operation*>(op.ptr), "");
    // Re-import op into the session to destroy it properly.
    ireeCompilerInvocationImportStealModule(invoke, op);
  }

  ireeCompilerInvocationDestroy(invoke);
  ireeCompilerSourceDestroy(source);
  ireeCompilerSessionDestroy(session);
  ireeCompilerGlobalShutdown();
  return 0;
}
