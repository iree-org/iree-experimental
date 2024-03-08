// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-prof-tools/graph-util.h"

#include <string>

#include "compiler/src/iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "compiler/src/iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "compiler/src/iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree-prof-tools/graph.h"
#include "third_party/abseil-cpp/absl/container/flat_hash_map.h"
#include "third_party/abseil-cpp/absl/log/check.h"
#include "third_party/abseil-cpp/absl/log/log.h"
#include "third_party/abseil-cpp/absl/status/status.h"
#include "third_party/abseil-cpp/absl/status/statusor.h"
#include "third_party/abseil-cpp/absl/strings/str_cat.h"
#include "third_party/abseil-cpp/absl/strings/string_view.h"
#include "third_party/llvm-project/llvm/include/llvm/Support/JSON.h"
#include "third_party/llvm-project/llvm/include/llvm/Support/raw_os_ostream.h"
#include "third_party/llvm-project/mlir/include/mlir/Dialect/Arith/IR/Arith.h"
#include "third_party/llvm-project/mlir/include/mlir/Dialect/Func/IR/FuncOps.h"
#include "third_party/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm-project/mlir/include/mlir/IR/Dialect.h"
#include "third_party/llvm-project/mlir/include/mlir/IR/Operation.h"

namespace iree_prof::graph {
namespace {

using namespace mlir::iree_compiler;

// A util template to return ASM-string of given T.
template <typename T>
std::string ToStr(T v) {
  std::stringstream ss;
  llvm::raw_os_ostream os(ss);
  v.print(os);
  os.flush();
  return ss.str();
}

// Returns integer from a contant mlir value.
int64_t ToInt(mlir::Value value) {
  auto constant_op = llvm::cast<mlir::arith::ConstantOp>(value.getDefiningOp());
  return llvm::cast<mlir::IntegerAttr>(constant_op.getValue()).getInt();
}

// Resource access flag of stream.cmd.dispatch op.
enum DispatchResourceAccess {
  ACCESS_READ = 1,
  ACCESS_WRITE = 2,
  ACCESS_READ_WRITE = 3,
};

// Returns resource access flag of stream.cmd.dispatch op.
DispatchResourceAccess ToAccess(mlir::Attribute attr) {
  return static_cast<DispatchResourceAccess>(
             llvm::cast<mlir::IntegerAttr>(attr).getInt());
}

// Returns a string used as label for stream.cmd.dispatch op.
std::string GetLabel(IREE::Stream::CmdDispatchOp op) {
  auto full_str = ToStr(op.getEntryPoints()[0]);
  auto pos = full_str.rfind("::@");
  if (pos != full_str.npos) {
    return full_str.substr(pos + 3);
  }
  return full_str;
}

// Returns a string used as label for stream.tensor.import op.
std::string GetLabel(IREE::Stream::TensorImportOp op) {
  auto source = llvm::cast<mlir::BlockArgument>(op.getSource());
  return absl::StrCat(ToStr(op->getName()), ":arg", source.getArgNumber());
}

// Returns a string used as label for util.global.load op.
std::string GetLabel(IREE::Util::GlobalLoadOp op) {
  return absl::StrCat(ToStr(op->getName()), ":", op.getGlobal().str());
}

// Builds a resource id with |index|, |offset| and |size|.
std::string GetResourceId(int index, int64_t offset, int64_t size) {
  return absl::StrCat("resource-", index, "[", offset, ":", offset + size, "]");
}

// Returns a function with most stream ops.
// Returns absl::NotFoundError if no functions have stream ops.
// TODO(byungchul): It might be better to find a public function with 1+ inputs
// of !hal.buffer_view type, not starting with "global$" as get/set functions.
// TODO(byungchul): Consider mlir::FailureOr<> instead of absl::StatusOr<>.
absl::StatusOr<mlir::func::FuncOp> GetFuncWithMostStreams(
    mlir::ModuleOp module) {
  int max_num_streams = 0;
  absl::StatusOr<mlir::func::FuncOp> result =
      absl::NotFoundError("Can't find a func with streams");

  module->walk([&](mlir::func::FuncOp func) {
    int num_streams = 0;
    func.walk([&num_streams](mlir::Operation* op) {
      if (llvm::isa<IREE::Stream::StreamDialect>(op->getDialect())) {
        ++num_streams;
      }
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

// Returns a function matched with function |name|.
// Returns absl::NotFoundError if no functions are matched.
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

// Adds a node with |label| and |parent_namespace| into |graph|.
GraphNode& AddNode(absl::string_view label,
                   absl::string_view parent_namespace,
                   Graph& graph) {
  graph.nodes.push_back(std::make_unique<GraphNode>());
  auto& node = *graph.nodes.back();
  node.id = absl::StrCat(graph.nodes.size() - 1);
  node.label = label;
  node.node_namespace = parent_namespace;
  return node;
}

// Adds an output of |source| as an incoming edge of |node|.
// If the same edge, i.e. with same |source| and |source_output_index| already
// exists, returns nullptr.
IncomingEdge* AddIncomingEdge(const GraphNode& source,
                              int source_output_index,
                              GraphNode& node) {
  const auto& source_node_output_id =
      source.outputs_metadata[source_output_index].id;
  for (const auto& e : node.incoming_edges) {
    if (e.source_node_id == source.id &&
        e.source_node_output_id == source_node_output_id) {
      return nullptr;
    }
  }

  IncomingEdge& input = node.incoming_edges.emplace_back();
  input.source_node_id = source.id;
  input.source_node_output_id = source_node_output_id;
  int input_index = node.incoming_edges.size() - 1;
  input.target_node_input_id = absl::StrCat("input-", input_index);
  input.metadata.emplace_back("index", absl::StrCat(input_index));
  return &input;
}

// Adds an output with some default metadata into |node|.
Metadata& AddOutputMetadata(GraphNode& node) {
  Metadata& output = node.outputs_metadata.emplace_back();
  int output_index = node.outputs_metadata.size() - 1;
  output.id = absl::StrCat("output-", output_index);
  output.attrs.emplace_back("index", absl::StrCat(output_index));
  return output;
}

// Adds a node for |op| into |graph|.
GraphNode& AddNodeForOperation(
    mlir::Operation* op,
    absl::string_view label,
    const absl::flat_hash_map<mlir::Operation*, std::string>& op_namespaces,
    Graph& graph) {
  CHECK(op_namespaces.contains(op->getParentOp()));
  return AddNode(label, op_namespaces.at(op->getParentOp()), graph);
}

// Adds a node for |op| with a default label, i.e. op's name into |graph|.
GraphNode& AddNodeForOperation(
    mlir::Operation* op,
    const absl::flat_hash_map<mlir::Operation*, std::string>& op_namespaces,
    Graph& graph) {
  return AddNodeForOperation(op, ToStr(op->getName()), op_namespaces, graph);
}

// Finds a node whose label is matched with |label_to_match|.
// TODO(byungchul): Consider a map if this function is called many times.
GraphNode* FindNodeForLabel(const Graph& graph,
                            absl::string_view label_to_match) {
  for (const auto& n : graph.nodes) {
    if (label_to_match == n->label) {
      return n.get();
    }
  }
  return nullptr;
}

// Records op name as a namespace for ops which are just containers from graph's
// point of view.
void AddNamespace(
    mlir::Operation* op,
    absl::flat_hash_map<mlir::Operation*, std::string>& op_namespaces) {
  CHECK(op_namespaces.contains(op->getParentOp()));
  int index = op_namespaces.size();
  op_namespaces[op] = absl::StrCat(op_namespaces[op->getParentOp()], "/",
                                   ToStr(op->getName()), "-", index);
}

// Adds a |resource| of stream.cmd.dispatch |op| as an output or an incoming
// edge of |node|.
void AddResourceAsOutputOrIncomingEdge(IREE::Stream::CmdDispatchOp op,
                                       int resource_index,
                                       const Graph& graph,
                                       GraphNode& node) {
  auto resource =
      llvm::cast<mlir::BlockArgument>(op.getResources()[resource_index]);
  int index = resource.getArgNumber();
  int64_t offset = ToInt(op.getResourceOffsets()[resource_index]);
  int64_t size = ToInt(op.getResourceLengths()[resource_index]);
  auto resource_id = GetResourceId(index, offset, size);

  auto access = ToAccess(op.getResourceAccesses()[resource_index]);
  if (access & ACCESS_WRITE) {
    auto& output = AddOutputMetadata(node);
    output.id = resource_id;
    output.attrs.emplace_back("arg_index", absl::StrCat(index));
    output.attrs.emplace_back("offset", absl::StrCat(offset));
    output.attrs.emplace_back("size", absl::StrCat(size));
    output.attrs.emplace_back("type", ToStr(resource.getType()));
    if (!(access & ACCESS_READ)) {
      return;
    }
  }

  CHECK(access & ACCESS_READ);

  // First, find the last node which has an output of the same resource id.
  // TODO(byungchul): Consider the control flow, e.g. scf or cf ops.
  for (auto it = graph.nodes.rbegin(); it != graph.nodes.rend(); ++it) {
    if (it->get() == &node) {
      continue;
    }
    for (int i = 0; i < (*it)->outputs_metadata.size(); ++i) {
      if ((*it)->outputs_metadata[i].id == resource_id) {
        AddIncomingEdge(**it, i, node);
        return;
      }
    }
  }

  // Find a node from the original operand of this argument.
  if (auto parent_op = llvm::dyn_cast<IREE::Stream::CmdExecuteOp>(
          resource.getOwner()->getParentOp())) {
    auto operand_op = parent_op.getResourceOperands()[index].getDefiningOp();
    std::string label_to_match;
    if (auto op = llvm::dyn_cast<IREE::Stream::TensorImportOp>(operand_op)) {
      label_to_match = GetLabel(op);
    } else if (auto op = llvm::dyn_cast<IREE::Util::GlobalLoadOp>(operand_op)) {
      label_to_match = GetLabel(op);
    }

    if (!label_to_match.empty()) {
      auto* matched = FindNodeForLabel(graph, label_to_match);
      if (matched) {
        AddIncomingEdge(*matched, 0, node);
        return;
      }
    }
  }

  LOG(WARNING) << "Can't find a matching output for resource: " << resource_id;
}

// Adds a node for a stream.cmd.dispatch op.
void AddNode(
    IREE::Stream::CmdDispatchOp op,
    const absl::flat_hash_map<mlir::Operation*, std::string>& op_namespaces,
    Graph& graph) {
  auto& node = AddNodeForOperation(op, GetLabel(op), op_namespaces, graph);
  for (int i = 0; i < op.getResources().size(); ++i) {
    AddResourceAsOutputOrIncomingEdge(op, i, graph, node);
  }
}

// Adds a node for a stream.cmd.fill op.
void AddNode(
    IREE::Stream::CmdFillOp op,
    const absl::flat_hash_map<mlir::Operation*, std::string>& op_namespaces,
    Graph& graph) {
  auto& node = AddNodeForOperation(op, op_namespaces, graph);
  auto target = llvm::cast<mlir::BlockArgument>(op.getTarget());
  int index = target.getArgNumber();
  int64_t offset = ToInt(op.getTargetOffset());
  int64_t size = ToInt(op.getTargetLength());
  auto resource_id = GetResourceId(index, offset, size);

  auto& output = AddOutputMetadata(node);
  output.id = resource_id;
  output.attrs.emplace_back("arg_index", absl::StrCat(index));
  output.attrs.emplace_back("offset", absl::StrCat(offset));
  output.attrs.emplace_back("size", absl::StrCat(size));
  output.attrs.emplace_back("type", ToStr(op.getTarget().getType()));
}

// Adds a node for a stream.tensor.import op.
void AddNode(
    IREE::Stream::TensorImportOp op,
    const absl::flat_hash_map<mlir::Operation*, std::string>& op_namespaces,
    Graph& graph) {
  auto& node = AddNodeForOperation(op, GetLabel(op), op_namespaces, graph);
  auto source = llvm::cast<mlir::BlockArgument>(op.getSource());
  AddIncomingEdge(*graph.nodes[0], source.getArgNumber(), node);
  auto& output = AddOutputMetadata(node);
  output.attrs.emplace_back("type", ToStr(op.getResultEncoding()));
}

// Adds a node for a stream.tensor.export op.
void AddNode(
    IREE::Stream::TensorExportOp op,
    const absl::flat_hash_map<mlir::Operation*, std::string>& op_namespaces,
    Graph& graph) {
  auto& node = AddNodeForOperation(op, op_namespaces, graph);
  auto& output = AddOutputMetadata(node);
  output.attrs.emplace_back("type", ToStr(op.getSourceEncoding()));

  // Connect to GraphOutput, graph.nodes[1].
  AddIncomingEdge(node, 0, *graph.nodes[1]);

  // Find a matching output with type for an incoming edge.
  // TODO(byungchul): Find matching output with source and index.
  auto type_to_match = ToStr(op.getSource().getType());
  for (auto it = graph.nodes.rbegin(); it != graph.nodes.rend(); ++it) {
    if (it->get() == &node) {
      continue;
    }
    for (int i = 0; i < (*it)->outputs_metadata.size(); ++i) {
      for (const auto& a : (*it)->outputs_metadata[i].attrs) {
        if (a.key == "type" && a.value == type_to_match) {
          AddIncomingEdge(**it, i, node);
          return;
        }
      }
    }
  }

  LOG(WARNING) << "Can't find a matching output: " << type_to_match;
}

// Adds a node for a util.global.load op.
void AddNode(
    IREE::Util::GlobalLoadOp op,
    const absl::flat_hash_map<mlir::Operation*, std::string>& op_namespaces,
    Graph& graph) {
  auto& node = AddNodeForOperation(op, GetLabel(op), op_namespaces, graph);
  AddOutputMetadata(node);
}

// Adds nodes accessed by |entrypoint| into |graph|.
absl::Status AddNodesForEntrypoint(mlir::func::FuncOp entrypoint,
                                   Graph& graph) {
  absl::flat_hash_map<mlir::Operation*, std::string> op_namespaces;
  op_namespaces[entrypoint] = entrypoint.getSymName().str();

  entrypoint->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    if (llvm::isa<IREE::Stream::CmdExecuteOp>(op) ||
        llvm::isa<IREE::Stream::CmdConcurrentOp>(op)) {
      AddNamespace(op, op_namespaces);
    } else if (auto cop = llvm::dyn_cast<IREE::Stream::CmdDispatchOp>(op)) {
      AddNode(cop, op_namespaces, graph);
    } else if (auto cop = llvm::dyn_cast<IREE::Stream::CmdFillOp>(op)) {
      AddNode(cop, op_namespaces, graph);
    } else if (auto cop = llvm::dyn_cast<IREE::Stream::TensorImportOp>(op)) {
      AddNode(cop, op_namespaces, graph);
    } else if (auto cop = llvm::dyn_cast<IREE::Stream::TensorExportOp>(op)) {
      AddNode(cop, op_namespaces, graph);
    } else if (auto cop = llvm::dyn_cast<IREE::Util::GlobalLoadOp>(op)) {
      AddNode(cop, op_namespaces, graph);
    } else if (llvm::isa<IREE::Stream::StreamDialect>(op->getDialect())) {
      if (!llvm::isa<IREE::Stream::YieldOp>(op) &&
          !llvm::isa<IREE::Stream::TimepointJoinOp>(op) &&
          !llvm::isa<IREE::Stream::TimepointAwaitOp>(op) &&
          !llvm::isa<IREE::Stream::ResourceAllocaOp>(op) &&
          !llvm::isa<IREE::Stream::ResourceDeallocaOp>(op)) {
        LOG(INFO) << "Ignore " << ToStr(op->getName());
      }
    }
  });

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<GraphCollection> GetGraphCollection(
    mlir::ModuleOp module,
    absl::string_view label,
    absl::string_view entrypoint) {
  auto func = entrypoint.empty() ? GetFuncWithMostStreams(module)
                                 : GetFuncWithName(module, entrypoint);
  if (!func.ok()) {
    return func.status();
  }

  GraphCollection collection;
  collection.label = label;
  Graph& graph = collection.graphs.emplace_back(func->getSymName().str());

  // Add graph input/output nodes.
  GraphNode& input_node = AddNode("GraphInput", "", graph);
  AddOutputMetadata(input_node);
  AddNode("GraphOutput", "", graph);
  // Graph outputs will be filled by entrypoint later.

  auto status = AddNodesForEntrypoint(*func, graph);
  if (!status.ok()) {
    return status;
  }

  return collection;
}

}  // namespace iree_prof::graph
