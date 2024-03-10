// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <filesystem>
#include <fstream>
#include <string>

#include "compiler/bindings/c/iree/compiler/embedding_api.h"
#include "compiler/bindings/c/iree/compiler/mlir_interop.h"
#include "iree-prof-tools/graph.h"
#include "iree-prof-tools/graph-util.h"
#include "third_party/abseil-cpp/absl/base/log_severity.h"
#include "third_party/abseil-cpp/absl/log/globals.h"
#include "third_party/abseil-cpp/absl/log/initialize.h"
#include "third_party/abseil-cpp/absl/log/log.h"
#include "third_party/abseil-cpp/absl/flags/flag.h"
#include "third_party/abseil-cpp/absl/flags/parse.h"
#include "third_party/llvm-project/llvm/include/llvm/Support/raw_os_ostream.h"
#include "third_party/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm-project/mlir/include/mlir/IR/Operation.h"

ABSL_FLAG(std::string, input_iree_file, "",
          "IREE MLIR text assembly file to read.");
ABSL_FLAG(std::string, output_json_file, "", "Graph JSON file to write.");
ABSL_FLAG(std::string, function, "",
          "Entrypoint function. If empty, pick one with most stream ops.");
ABSL_FLAG(int, min_num_ops_to_group, 3,
          "Minimum number of operations to group in a namespace when same "
          "operations continue in a row. If <0, grouping is diabled.");

namespace iree_prof::graph {
namespace {

void ConvertToGraphJson(mlir::ModuleOp module) {
  std::string output_json_file = absl::GetFlag(FLAGS_output_json_file);
  std::ofstream fout(output_json_file.c_str());
  if (!fout.is_open() || !fout.good()) {
    LOG(ERROR) << "Can't open output file: " << output_json_file;
    return;
  }

  auto label = std::filesystem::path(absl::GetFlag(FLAGS_input_iree_file))
               .filename().string();
  auto graph_collection =
      GetGraphCollection(module, label, absl::GetFlag(FLAGS_function),
                         absl::GetFlag(FLAGS_min_num_ops_to_group));
  if (!graph_collection.ok()) {
    LOG(ERROR) << graph_collection.status();
    return;
  }

  {
    llvm::raw_os_ostream os(fout);
    os << graph_collection->Json();
  }

  LOG(INFO) << "Wrote " << fout.tellp() << " bytes to " << output_json_file;
  fout.close();
}

}  // namespace
}  // namespace iree_prof::graph

int main(int argc, char** argv) {
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  absl::InitializeLog();
  absl::ParseCommandLine(argc, argv);

  // TODO(byungchul): Load iree/mlir dialects directly and reduce the dependency
  // on iree compiler.
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
  LOG(INFO) << "Parsing " << input_iree_file << "...";
  if (ireeCompilerInvocationParseSource(invoke, source)) {
    LOG(INFO) << "Parsed " << input_iree_file << " successfully.";
    auto op = ireeCompilerInvocationExportStealModule(invoke);
    iree_prof::graph::ConvertToGraphJson(
        llvm::cast<mlir::ModuleOp>(reinterpret_cast<mlir::Operation*>(op.ptr)));
    // Re-import op into the session to destroy it properly.
    ireeCompilerInvocationImportStealModule(invoke, op);
  }

  ireeCompilerInvocationDestroy(invoke);
  ireeCompilerSourceDestroy(source);
  ireeCompilerSessionDestroy(session);
  ireeCompilerGlobalShutdown();
  return 0;
}
