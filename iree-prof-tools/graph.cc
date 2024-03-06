// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-prof-tools/graph.h"

#include "third_party/llvm-project/llvm/include/llvm/Support/JSON.h"

namespace iree_prof::graph {

llvm::json::Object Attribute::Json() {
  llvm::json::Object json_attr;
  json_attr["key"] = key;
  json_attr["value"] = value;
  return json_attr;
}

llvm::json::Object Metadata::Json() {
  llvm::json::Object json_metadata;
  json_metadata["id"] = id;
  json_metadata["attrs"] = llvm::json::Array();
  llvm::json::Array* json_attrs = json_metadata["attrs"].getAsArray();
  for (Attribute& attr : attrs) {
    json_attrs->push_back(attr.Json());
  }
  return json_metadata;
}

llvm::json::Object GraphEdge::Json() {
  llvm::json::Object json_edge;
  json_edge["sourceNodeId"] = source_node_id;
  json_edge["sourceNodeOutputId"] = source_node_output_id;
  json_edge["targetNodeInputId"] = target_node_input_id;
  json_edge["edgeMetadata"] = llvm::json::Array();
  llvm::json::Array* json_attrs = json_edge["edgeMetadata"].getAsArray();
  for (Attribute& attr : edge_metadata) {
    json_attrs->push_back(attr.Json());
  }
  return json_edge;
}

llvm::json::Object GraphNode::Json() {
  llvm::json::Object json_node;
  json_node["id"] = node_id;
  json_node["label"] = node_label;
  json_node["namespace"] = node_name;
  json_node["subgraphIds"] = subgraph_ids;

  json_node["attrs"] = llvm::json::Array();
  llvm::json::Array* json_attrs = json_node["attrs"].getAsArray();
  for (Attribute& attr : node_attrs) {
    json_attrs->push_back(attr.Json());
  }

  json_node["incomingEdges"] = llvm::json::Array();
  llvm::json::Array* json_edges = json_node["incomingEdges"].getAsArray();
  for (GraphEdge& edge : incoming_edges) {
    json_edges->push_back(edge.Json());
  }

  json_node["inputsMetadata"] = llvm::json::Array();
  llvm::json::Array* json_inputs_metadata =
      json_node["inputsMetadata"].getAsArray();
  for (Metadata& metadata : inputs_metadata) {
    json_inputs_metadata->push_back(metadata.Json());
  }

  json_node["outputsMetadata"] = llvm::json::Array();
  llvm::json::Array* json_outputs_metadata =
      json_node["outputsMetadata"].getAsArray();
  for (Metadata& metadata : outputs_metadata) {
    json_outputs_metadata->push_back(metadata.Json());
  }
  return json_node;
}

llvm::json::Object Graph::Json() {
  llvm::json::Object json_graph;
  json_graph["id"] = graph_id;
  json_graph["nodes"] = llvm::json::Array();
  llvm::json::Array* json_nodes = json_graph["nodes"].getAsArray();
  for (GraphNode& node : nodes) {
    json_nodes->push_back(node.Json());
  }
  return json_graph;
}

llvm::json::Object GraphCollection::Json() {
  llvm::json::Object json_graph;
  json_graph["label"] = label;
  json_graph["graphs"] = llvm::json::Array();
  llvm::json::Array* json_graphs = json_graph["graphs"].getAsArray();
  for (Graph& graph : graphs) {
    json_graphs->push_back(graph.Json());
  }
  return json_graph;
}

}  // namespace iree_prof::graph
