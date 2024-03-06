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
#include "compiler/src/iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "compiler/src/iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree-prof-tools/graph.h"
#include "third_party/abseil-cpp/absl/base/log_severity.h"
#include "third_party/abseil-cpp/absl/log/globals.h"
#include "third_party/abseil-cpp/absl/log/initialize.h"
#include "third_party/abseil-cpp/absl/log/log.h"
#include "third_party/abseil-cpp/absl/flags/flag.h"
#include "third_party/abseil-cpp/absl/flags/parse.h"
#include "third_party/abseil-cpp/absl/status/status.h"
#include "third_party/abseil-cpp/absl/status/statusor.h"
#include "third_party/abseil-cpp/absl/strings/str_cat.h"
#include "third_party/llvm-project/llvm/include/llvm/Support/JSON.h"
#include "third_party/llvm-project/llvm/include/llvm/Support/raw_os_ostream.h"
#include "third_party/llvm-project/mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "third_party/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm-project/mlir/include/mlir/IR/Dialect.h"
#include "third_party/llvm-project/mlir/include/mlir/IR/Operation.h"

ABSL_FLAG(std::string, input_iree_file, "",
          "IREE MLIR text assembly file to read.");
ABSL_FLAG(std::string, output_json_file, "", "Graph JSON file to write.");
ABSL_FLAG(std::string, function, "",
          "Entrypoint function. If empty, pick one with most stream ops.");

namespace iree_prof {
namespace {

absl::StatusOr<mlir::func::FuncOp> GetFuncWithMostStreams(
    mlir::ModuleOp module) {
  int max_num_streams = 0;
  absl::StatusOr<mlir::func::FuncOp> result =
      absl::NotFoundError("Can't find a func with streams");

  module->walk([&](mlir::func::FuncOp func) {
    int num_streams = 0;
    func.walk([&num_streams](
        mlir::iree_compiler::IREE::Stream::CmdExecuteOp exec) {
      exec.walk([&num_streams](mlir::Operation* op) {
        if (llvm::isa<mlir::iree_compiler::IREE::Stream::StreamDialect>(
                op->getDialect())) {
          ++num_streams;
        }
      });
    });

    if (num_streams > max_num_streams) {
      max_num_streams = num_streams;
      result = func;
    }
  });

  if (result.ok()) {
    LOG(INFO) << "Func with max streams = "  << result->getSymName().str()
              << ", # of streams = " << max_num_streams;
  }
  return result;
}

absl::StatusOr<mlir::func::FuncOp> GetFuncWithName(mlir::ModuleOp module,
                                                   absl::string_view name) {
  absl::StatusOr<mlir::func::FuncOp> result = absl::NotFoundError(
      absl::StrCat("Can't find a func of name \"", name, "\""));

  module->walk([&](mlir::func::FuncOp func) -> mlir::WalkResult {
    if (name == func.getSymName().str()) {
      result = func;
      return mlir::WalkResult::skip();
    }
    return mlir::WalkResult::advance();
  });

  if (result.ok()) {
    LOG(INFO) << "Found a func with name = " << result->getSymName().str();
  }
  return result;
}

absl::Status AddNode(mlir::func::FuncOp func, graph::Graph& graph) {
  graph.nodes.emplace_back();
  graph::GraphNode& node = graph.nodes.back();
  node.node_id = func.getSymName().str();
  node.node_label = func.getSymName().str();
  node.node_name = func.getSymName().str();
  return absl::OkStatus();
}

absl::StatusOr<graph::GraphCollection> GetGraphCollection(
    mlir::Operation* op,
    absl::string_view label) {
  if (!llvm::isa<mlir::ModuleOp>(op)) {
    return absl::InvalidArgumentError("Given module is not valid");
  }

  auto module = llvm::cast<mlir::ModuleOp>(op);

  std::string entrypoint = absl::GetFlag(FLAGS_function);
  auto func = entrypoint.empty() ? GetFuncWithMostStreams(module)
                                 : GetFuncWithName(module, entrypoint);
  if (!func.ok()) {
    return func.status();
  }

  graph::GraphCollection collection;
  collection.label = label;
  collection.graphs.emplace_back(func->getSymName().str());
  auto status = AddNode(*func, collection.graphs.back());
  if (!status.ok()) {
    return status;
  }
  return collection;
}

void ConvertToGraphJson(mlir::Operation* op) {
  std::string output_json_file = absl::GetFlag(FLAGS_output_json_file);
  std::ofstream fout(output_json_file.c_str());
  if (!fout.is_open() || !fout.good()) {
    LOG(ERROR) << "Can't open output file: " << output_json_file;
    return;
  }

  auto label = std::filesystem::path(absl::GetFlag(FLAGS_input_iree_file))
               .filename().string();
  auto graph_collection = GetGraphCollection(op, label);
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
}  // namespace iree_prof

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
  if (ireeCompilerInvocationParseSource(invoke, source)) {
    LOG(INFO) << "Parsing " << input_iree_file << " done successfully.";
    auto op = ireeCompilerInvocationExportStealModule(invoke);
    iree_prof::ConvertToGraphJson(reinterpret_cast<mlir::Operation*>(op.ptr));
    // Re-import op into the session to destroy it properly.
    ireeCompilerInvocationImportStealModule(invoke, op);
  }

  ireeCompilerInvocationDestroy(invoke);
  ireeCompilerSourceDestroy(source);
  ireeCompilerSessionDestroy(session);
  ireeCompilerGlobalShutdown();
  return 0;
}
