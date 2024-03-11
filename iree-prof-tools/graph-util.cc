// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-prof-tools/graph-util.h"

#include <string>
#include <type_traits>

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
#include "third_party/llvm-project/mlir/include/mlir/IR/AsmState.h"
#include "third_party/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"
#include "third_party/llvm-project/mlir/include/mlir/IR/Dialect.h"
#include "third_party/llvm-project/mlir/include/mlir/IR/Operation.h"

namespace iree_prof::graph {
namespace {

using namespace mlir::iree_compiler;

// Printing flag to avoid unnecessary verifications in ToStr(). Verification
// slows down the process due to thread synchronizations.
mlir::OpPrintingFlags GetPrintingFlags() {
  return mlir::OpPrintingFlags().assumeVerified().useLocalScope();
}

// Helper templates for ToStr() below.
// Print() for non-mlir::Value types.
template <typename T,
          std::enable_if_t<!std::is_base_of_v<mlir::Value, T>, bool> = true>
void Print(T v, llvm::raw_os_ostream& os) {
  mlir::AsmState as(v.getContext(), GetPrintingFlags());
  v.print(os, as);
}

// Print() for mlir::Value and its subclasses.
template <typename T,
          std::enable_if_t<std::is_base_of_v<mlir::Value, T>, bool> = true>
void Print(T v, llvm::raw_os_ostream& os) {
  v.printAsOperand(os, GetPrintingFlags());
}

// Print() for mlir::OperationName which doesn't have print() either with
// AmsState or with OpPrintingFlags.
template <>
void Print<mlir::OperationName>(mlir::OperationName n,
                                llvm::raw_os_ostream& os) {
  n.print(os);
}

// A util templates to return ASM-string of given T.
template <typename T>
std::string ToStr(T v) {
  std::stringstream ss;
  llvm::raw_os_ostream os(ss);
  Print<T>(v, os);
  os.flush();
  return ss.str();
}

std::string ToTypeStr(mlir::Type type, mlir::Value size) {
  return absl::StrCat(ToStr(type), "{", ToStr(size), "}");
}

// Returns a graph KeyValuePair for a value which is either a constant value or
// a variable.
KeyValuePair ToKeyValuePair(absl::string_view key, mlir::Value val) {
  if (auto op = llvm::dyn_cast<mlir::arith::ConstantOp>(val.getDefiningOp())) {
    return KeyValuePair(key,
                        llvm::cast<mlir::IntegerAttr>(op.getValue()).getInt());
  }

  return KeyValuePair(key, ToStr(val));
}

// Resource access flag of stream.cmd.dispatch op.
enum DispatchResourceAccess {
  ACCESS_READ = 1,
  ACCESS_WRITE = 2,
  ACCESS_READ_WRITE = ACCESS_READ|ACCESS_WRITE,
};

// Returns resource access flag of stream.cmd.dispatch op.
DispatchResourceAccess ToAccess(mlir::Attribute attr) {
  return static_cast<DispatchResourceAccess>(
             llvm::cast<mlir::IntegerAttr>(attr).getInt());
}

// Builds a block argument id with |index|, |offset| and |length|.
std::string GetArgId(int index,
                     const KeyValuePair& offset,
                     const KeyValuePair& length) {
  return absl::StrCat("arg", index, "[", offset.value, ":",
                      (offset.int_value && length.int_value
                       ? absl::StrCat(*offset.int_value + *length.int_value)
                       : absl::StrCat(offset.value, "+", length.value)), "]");
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

// Returns a string used as label for stream.cmd.copy op.
std::string GetLabel(IREE::Stream::CmdCopyOp op) {
  auto source_arg = llvm::cast<mlir::BlockArgument>(op.getSource());
  auto target_arg = llvm::cast<mlir::BlockArgument>(op.getTarget());
  auto source_id = GetArgId(source_arg.getArgNumber(),
                            ToKeyValuePair("offset", op.getSourceOffset()),
                            ToKeyValuePair("length", op.getLength()));
  auto target_id = GetArgId(target_arg.getArgNumber(),
                            ToKeyValuePair("offset", op.getTargetOffset()),
                            ToKeyValuePair("length", op.getLength()));
  return absl::StrCat(ToStr(op->getName()), ":", source_id, "->", target_id);
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

// Whether an op is of function, i.e. func.func or util.func.
bool IsFunction(mlir::Operation* op) {
  return llvm::isa<mlir::func::FuncOp>(op) ||
         llvm::isa<IREE::Util::FuncOp>(op);
}

// Returns the function name. IsFunction(op) must return true.
std::string GetFuncName(mlir::Operation* op) {
  if (auto func = llvm::dyn_cast<mlir::func::FuncOp>(op)) {
    return func.getSymName().str();
  }
  return llvm::dyn_cast<IREE::Util::FuncOp>(op).getSymName().str();
}

// Returns a function with most stream ops.
// Returns absl::NotFoundError if no functions have stream ops.
// TODO(byungchul): It might be better to find a public function with 1+ inputs
// of !hal.buffer_view type, not starting with "global$" as get/set functions.
// TODO(byungchul): Consider mlir::FailureOr<> instead of absl::StatusOr<>.
absl::StatusOr<mlir::Operation*> GetFuncWithMostStreams(mlir::ModuleOp module) {
  int max_num_streams = 0;
  absl::StatusOr<mlir::Operation*> result =
      absl::NotFoundError("Can't find a func with streams");

  module->walk([&](mlir::Operation* op) {
    if (!IsFunction(op)) {
      return;
    }

    int num_streams = 0;
    op->walk([&num_streams](mlir::Operation* child_op) {
      if (llvm::isa<IREE::Stream::StreamDialect>(child_op->getDialect())) {
        ++num_streams;
      }
    });

    if (num_streams > max_num_streams) {
      max_num_streams = num_streams;
      result = op;
    }
  });

  if (result.ok()) {
    LOG(INFO) << "Func with max streams = "  << GetFuncName(*result)
              << ", # of streams = " << max_num_streams;
  }
  return result;
}

// Returns a function matched with function |name|.
// Returns absl::NotFoundError if no functions are matched.
absl::StatusOr<mlir::Operation*> GetFuncWithName(mlir::ModuleOp module,
                                                 absl::string_view name) {
  absl::StatusOr<mlir::Operation*> result = absl::NotFoundError(
      absl::StrCat("Can't find a func of name \"", name, "\""));

  module->walk([&](mlir::Operation* op) -> mlir::WalkResult {
    if (IsFunction(op)) {
      if (name == GetFuncName(op)) {
        result = op;
        return mlir::WalkResult::skip();
      }
    }
    return mlir::WalkResult::advance();
  });

  if (result.ok()) {
    LOG(INFO) << "Found a func with name = " << GetFuncName(*result);
  }
  return result;
}

// Adds a node with |label| and |parent_namespace| into |graph|.
GraphNode& AddNode(absl::string_view label,
                   absl::string_view parent_namespace,
                   absl::string_view op_name,
                   Graph& graph) {
  graph.nodes.push_back(std::make_unique<GraphNode>());
  auto& node = *graph.nodes.back();
  node.id = absl::StrCat(graph.nodes.size() - 1);
  node.label = label;
  node.node_namespace = parent_namespace;
  node.node_attrs.emplace_back("op", op_name);
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
  input.metadata.emplace_back("input_index", input_index);
  return &input;
}

// Adds an output with some default metadata into |node|.
Metadata& AddOutputMetadata(GraphNode& node) {
  Metadata& output = node.outputs_metadata.emplace_back();
  int output_index = node.outputs_metadata.size() - 1;
  output.id = absl::StrCat("output-", output_index);
  output.attrs.emplace_back("node_id", node.id);
  output.attrs.emplace_back("output_index", output_index);
  return output;
}

// Adds a node for |op| into |graph|.
GraphNode& AddNodeForOperation(
    mlir::Operation* op,
    absl::string_view label,
    const absl::flat_hash_map<mlir::Operation*, std::string>& op_namespaces,
    Graph& graph) {
  CHECK(op_namespaces.contains(op->getParentOp()));
  return AddNode(label, op_namespaces.at(op->getParentOp()),
                 ToStr(op->getName()), graph);
}

// Adds a node for |op| with a default label, i.e. op's name into |graph|.
GraphNode& AddNodeForOperation(
    mlir::Operation* op,
    const absl::flat_hash_map<mlir::Operation*, std::string>& op_namespaces,
    Graph& graph) {
  return AddNodeForOperation(op, ToStr(op->getName()), op_namespaces, graph);
}

// Finds a node whose label is matched with |label_to_match| in reverse order.
// TODO(byungchul): Consider a map if this function is called many times.
GraphNode* FindNodeForLabel(const Graph& graph,
                            absl::string_view label_to_match,
                            const GraphNode& node_to_skip) {
  for (auto it = graph.nodes.rbegin(); it != graph.nodes.rend(); ++it) {
    if (it->get() != &node_to_skip && (*it)->label == label_to_match) {
      return it->get();
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

// Whether a node is in a stream.cmd.concurrent op, i.e. last part of
// namespace contains "stream.cmd.concurrent".
bool IsInConcurrent(const GraphNode& node) {
  auto pos = node.node_namespace.rfind('/');
  if (pos == node.node_namespace.npos) {
    pos = 0;
  }
  return node.node_namespace.find("stream.cmd.concurrent", pos) !=
         node.node_namespace.npos;
}

// Whether two nodes are in a stream.cmd.concurrent op.
bool IsInSameConcurrent(const GraphNode& a, const GraphNode& b) {
  return a.node_namespace == b.node_namespace && IsInConcurrent(a);
}

// Returns an output metadata attribute matched with key.
const KeyValuePair* GetOutputAttribute(const Metadata& output,
                                       absl::string_view key) {
  for (const auto& a : output.attrs) {
    if (key == a.key) {
      return &a;
    }
  }
  return nullptr;
}

// Whether an output is matched with |arg_id| or a tuple of |index|, |offset|,
// and |length|.
bool IsOutputMatched(const Metadata& output,
                     absl::string_view arg_id,
                     int index,
                     const KeyValuePair& offset,
                     const KeyValuePair& length) {
  if (output.id == arg_id) {
    return true;
  }
  if (auto output_index = GetOutputAttribute(output, "arg_index")) {
    if (output_index->int_value && *output_index->int_value == index) {
      // Output doesn't have to be matched with [offset:offset+length] exactly.
      // As long as they are overlapped, it can be assumed the output is used
      // as an input resource. It is NOT overlapped only when the end of either
      // one is <= the start of the other. Note that start is inclusive while
      // the end is not.
      auto output_offset = GetOutputAttribute(output, "offset");
      if (output_offset) {
        // Quick return when 2 offsets (either int value or arg name) are same.
        if (offset.value == output_offset->value) {
          return true;
        }
        auto output_length = GetOutputAttribute(output, "length");
        if (output_length && offset.int_value && length.int_value &&
            output_offset->int_value && output_length->int_value) {
          auto start = *offset.int_value;
          auto end = start + *length.int_value;
          CHECK_LE(start, end);
          auto output_start = *output_offset->int_value;
          auto output_end = output_start + *output_length->int_value;
          CHECK_LE(output_start, output_end);
          if (end <= output_start || start >= output_end) {
            return false;
          }
          return true;
        }
      }
    }
  }
  return false;
}

// Adds a block argument as an output of |node|.
void AddArgAsOutput(mlir::BlockArgument arg,
                    mlir::Value arg_offset,
                    mlir::Value arg_length,
                    mlir::Value arg_size,
                    GraphNode& node) {
  auto offset = ToKeyValuePair("offset", arg_offset);
  auto length = ToKeyValuePair("length", arg_length);
  auto& output = AddOutputMetadata(node);
  output.id = GetArgId(arg.getArgNumber(), offset, length);
  output.attrs.emplace_back("arg_id", output.id);
  output.attrs.emplace_back("arg_index", arg.getArgNumber());
  output.attrs.emplace_back(std::move(offset));
  output.attrs.emplace_back(std::move(length));
  output.attrs.emplace_back("type", ToTypeStr(arg.getType(), arg_size));
}

// Adds a block argument as an incoming edge of |node|.
bool AddArgAsIncomingEdge(mlir::BlockArgument arg,
                          mlir::Value arg_offset,
                          mlir::Value arg_length,
                          const Graph& graph,
                          GraphNode& node) {
  auto offset = ToKeyValuePair("offset", arg_offset);
  auto length = ToKeyValuePair("length", arg_length);
  auto arg_id = GetArgId(arg.getArgNumber(), offset, length);

  // First, find the last node(s) which is not in the same stream.cmd.concurrent
  // and has an output matched with the argument. Note that they could be more
  // than one if last nodes are in a stream.cmd.concurrent (which is different
  // than that of this node).
  // TODO(byungchul): Consider the control flow, e.g. scf or cf ops. Uses in
  // Operation may have useful information.
  const GraphNode* matched = nullptr;
  for (auto it = graph.nodes.rbegin(); it != graph.nodes.rend(); ++it) {
    if (it->get() == &node || IsInSameConcurrent(**it, node)) {
      continue;
    }
    if (matched != nullptr && !IsInSameConcurrent(**it, *matched)) {
      // No more matched nodes in the same concurrent.
      return true;
    }
    for (int i = 0; i < (*it)->outputs_metadata.size(); ++i) {
      if (IsOutputMatched((*it)->outputs_metadata[i], arg_id,
                          arg.getArgNumber(), offset, length)) {
        matched = it->get();
        AddIncomingEdge(*matched, i, node);
        break;
      }
    }
  }

  // If not found, find a node from the original operand of this argument.
  if (auto parent_op = llvm::dyn_cast<IREE::Stream::CmdExecuteOp>(
          arg.getOwner()->getParentOp())) {
    auto operand_op =
        parent_op.getResourceOperands()[arg.getArgNumber()].getDefiningOp();
    std::string label_to_match;
    if (auto op = llvm::dyn_cast<IREE::Stream::TensorImportOp>(operand_op)) {
      label_to_match = GetLabel(op);
    } else if (auto op = llvm::dyn_cast<IREE::Util::GlobalLoadOp>(operand_op)) {
      label_to_match = GetLabel(op);
    }

    if (!label_to_match.empty()) {
      auto* matched = FindNodeForLabel(graph, label_to_match, node);
      if (matched) {
        AddIncomingEdge(*matched, 0, node);
        return true;
      }
    }
  }

  return false;
}

// Adds a node for a stream.cmd.dispatch op.
void AddNode(
    IREE::Stream::CmdDispatchOp op,
    const absl::flat_hash_map<mlir::Operation*, std::string>& op_namespaces,
    Graph& graph) {
  auto& node = AddNodeForOperation(op, GetLabel(op), op_namespaces, graph);
  for (int i = 0; i < op.getResources().size(); ++i) {
    auto arg = llvm::cast<mlir::BlockArgument>(op.getResources()[i]);
    auto access = ToAccess(op.getResourceAccesses()[i]);
    CHECK(access & ACCESS_READ_WRITE);  // Read, write or both must be set.

    if (access & ACCESS_WRITE) {
      AddArgAsOutput(arg, op.getResourceOffsets()[i],
                     op.getResourceLengths()[i], op.getResourceSizes()[i],
                     node);
    }

    if (access & ACCESS_READ) {
      if (!AddArgAsIncomingEdge(arg, op.getResourceOffsets()[i],
                                op.getResourceLengths()[i], graph, node)) {
        auto arg_id = GetArgId(
            arg.getArgNumber(),
            ToKeyValuePair("offset", op.getResourceOffsets()[i]),
            ToKeyValuePair("length", op.getResourceLengths()[i]));
        LOG(WARNING) << "Can't find a matching output: node=" << node.label
                     << ", id=" << node.id << ", arg=" << arg_id;
      }
    }
  }
}

// Adds a node for a stream.cmd.fill op.
void AddNode(
    IREE::Stream::CmdFillOp op,
    const absl::flat_hash_map<mlir::Operation*, std::string>& op_namespaces,
    Graph& graph) {
  auto& node = AddNodeForOperation(op, op_namespaces, graph);
  AddArgAsOutput(llvm::cast<mlir::BlockArgument>(op.getTarget()),
                 op.getTargetOffset(), op.getTargetLength(), op.getTargetSize(),
                 node);
}

// Adds an incoming edge from an output matched with |type| and |size|.
bool AddIncomingEdgeForType(mlir::Type type,
                            mlir::Value size,
                            const Graph& graph,
                            GraphNode& node) {
  auto type_to_match = ToTypeStr(type, size);
  for (auto it = graph.nodes.rbegin(); it != graph.nodes.rend(); ++it) {
    if (it->get() == &node || IsInSameConcurrent(**it, node)) {
      continue;
    }
    for (int i = 0; i < (*it)->outputs_metadata.size(); ++i) {
      if (auto* attr = GetOutputAttribute((*it)->outputs_metadata[i], "type")) {
        if (attr->value == type_to_match) {
          AddIncomingEdge(**it, i, node);
          return true;
        }
      }
    }
  }
  return false;
}

// Adds a node for a stream.cmd.copy op.
void AddNode(
    IREE::Stream::CmdCopyOp op,
    const absl::flat_hash_map<mlir::Operation*, std::string>& op_namespaces,
    Graph& graph) {
  auto& node = AddNodeForOperation(op, GetLabel(op), op_namespaces, graph);
  AddArgAsOutput(llvm::cast<mlir::BlockArgument>(op.getTarget()),
                 op.getTargetOffset(), op.getLength(), op.getTargetSize(),
                 node);

  if (!AddArgAsIncomingEdge(llvm::cast<mlir::BlockArgument>(op.getSource()),
                            op.getSourceOffset(), op.getLength(), graph, node)
      && !AddIncomingEdgeForType(op.getSource().getType(), op.getSourceSize(),
                                 graph, node)) {
    auto source_id = GetArgId(
        llvm::cast<mlir::BlockArgument>(op.getSource()).getArgNumber(),
        ToKeyValuePair("offset", op.getSourceOffset()),
        ToKeyValuePair("length", op.getLength()));
    LOG(WARNING) << "Can't find a matching output: node=" << node.label
                 << ", id=" << node.id << ", source=" << source_id << ", type="
                 << ToTypeStr(op.getSource().getType(), op.getSourceSize());
  }
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
  if (!AddIncomingEdgeForType(op.getSource().getType(), op.getSourceSize(),
                              graph, node)) {
    LOG(WARNING) << "Can't find a matching output: node=" << node.label
                 << ", type="
                 << ToTypeStr(op.getSource().getType(), op.getSourceSize());
  }
}

// Adds a node for a util.global.load op.
void AddNode(
    IREE::Util::GlobalLoadOp op,
    const absl::flat_hash_map<mlir::Operation*, std::string>& op_namespaces,
    Graph& graph) {
  auto& node = AddNodeForOperation(op, GetLabel(op), op_namespaces, graph);
  AddOutputMetadata(node);
}

// Append another namespace to group the operations if same operations continue
// too many times in a row.
// TODO(byungchul): Consider more groupings with compiler's help, or define
// graph IR for easier conversions.
void AppendNamespacesIfNecessary(const Graph& graph,
                                 int min_num_ops_to_group) {
  int index_of_first_same_op = 0;
  int num_nodes_of_same_op = 0;
  for (int i = 0; i < graph.nodes.size(); ++i) {
    const auto& node = *graph.nodes[i];
    CHECK_EQ(node.node_attrs[0].key, "op");
    const auto& first = *graph.nodes[index_of_first_same_op];
    const auto& op_name = first.node_attrs[0].value;
    if (node.node_attrs[0].value == op_name &&
        node.node_namespace == first.node_namespace) {
      ++num_nodes_of_same_op;
      continue;
    }

    // Append another namespace if there are too many same ops but they are NOT
    // already grouped in a stream.cmd.concurrent.
    if (num_nodes_of_same_op >= min_num_ops_to_group &&
        !IsInConcurrent(first)) {
      auto suffix = absl::StrCat("/", op_name, "-", first.id);
      for (int i = 0; i < num_nodes_of_same_op; ++i) {
        graph.nodes[index_of_first_same_op + i]->node_namespace.append(suffix);
      }
    }

    num_nodes_of_same_op = 1;
    index_of_first_same_op = i;
  }
}

// Adds nodes accessed by |entrypoint| into |graph|.
absl::Status AddNodesForEntrypoint(mlir::Operation* entrypoint,
                                   int min_num_ops_to_group,
                                   Graph& graph) {
  absl::flat_hash_map<mlir::Operation*, std::string> op_namespaces;
  op_namespaces[entrypoint] = GetFuncName(entrypoint);

  absl::flat_hash_map<std::string, int> ops_ignored;
  entrypoint->walk<mlir::WalkOrder::PreOrder>([&](mlir::Operation* op) {
    if (llvm::isa<IREE::Stream::CmdExecuteOp>(op) ||
        llvm::isa<IREE::Stream::CmdConcurrentOp>(op)) {
      AddNamespace(op, op_namespaces);
    } else if (auto cop = llvm::dyn_cast<IREE::Stream::CmdDispatchOp>(op)) {
      AddNode(cop, op_namespaces, graph);
    } else if (auto cop = llvm::dyn_cast<IREE::Stream::CmdFillOp>(op)) {
      AddNode(cop, op_namespaces, graph);
    } else if (auto cop = llvm::dyn_cast<IREE::Stream::CmdCopyOp>(op)) {
      AddNode(cop, op_namespaces, graph);
    } else if (auto cop = llvm::dyn_cast<IREE::Stream::TensorImportOp>(op)) {
      AddNode(cop, op_namespaces, graph);
    } else if (auto cop = llvm::dyn_cast<IREE::Stream::TensorExportOp>(op)) {
      AddNode(cop, op_namespaces, graph);
    } else if (auto cop = llvm::dyn_cast<IREE::Util::GlobalLoadOp>(op)) {
      AddNode(cop, op_namespaces, graph);
    } else {
      ops_ignored[ToStr(op->getName())]++;
    }
  });

  if (min_num_ops_to_group >= 0) {
    AppendNamespacesIfNecessary(graph, min_num_ops_to_group);
  }

  for (const auto& it : ops_ignored) {
    LOG(WARNING) << "Ignored " << it.second << " " << it.first << "(s).";
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<GraphCollection> GetGraphCollection(
    mlir::ModuleOp module,
    absl::string_view label,
    absl::string_view entrypoint,
    int min_num_ops_to_group) {
  auto func = entrypoint.empty() ? GetFuncWithMostStreams(module)
                                 : GetFuncWithName(module, entrypoint);
  if (!func.ok()) {
    return func.status();
  }

  GraphCollection collection;
  collection.label = label;
  Graph& graph = collection.graphs.emplace_back(GetFuncName(*func));

  // Add graph input/output nodes.
  GraphNode& input_node = AddNode("GraphInput", "", "graph-input", graph);
  AddOutputMetadata(input_node);
  AddNode("GraphOutput", "", "graph-output", graph);
  // Graph outputs will be filled by entrypoint later.

  auto status = AddNodesForEntrypoint(*func, min_num_ops_to_group, graph);
  if (!status.ok()) {
    return status;
  }

  return collection;
}

}  // namespace iree_prof::graph
