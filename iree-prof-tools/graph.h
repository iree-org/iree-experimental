// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PROF_GRAPH_H_
#define IREE_PROF_GRAPH_H_

#include <string>

#include "third_party/llvm-project/llvm/include/llvm/Support/JSON.h"

namespace iree_prof::graph {

struct Attribute {
  Attribute(std::string key, std::string value)
      : key(std::move(key)), value(std::move(value)) {}
  std::string key;
  std::string value;

  llvm::json::Object Json();
};

struct Metadata {
  std::string id;
  std::vector<Attribute> attrs;

  llvm::json::Object Json();
};

struct GraphEdge {
  std::string source_node_id;
  std::string source_node_output_id;
  std::string target_node_input_id;
  std::vector<Attribute> edge_metadata;

  llvm::json::Object Json();
};

struct GraphNode {
  std::string node_id;
  std::string node_label;
  std::string node_name;
  std::vector<std::string> subgraph_ids;
  std::vector<Attribute> node_attrs;
  std::vector<GraphEdge> incoming_edges;
  std::vector<Metadata> inputs_metadata;
  std::vector<Metadata> outputs_metadata;

  llvm::json::Object Json();
};

struct Graph {
  explicit Graph(std::string graph_id)
      : graph_id(std::move(graph_id)) {}
  std::string graph_id;
  std::vector<GraphNode> nodes;

  llvm::json::Object Json();
};

struct GraphCollection {
  std::string label;
  std::vector<Graph> graphs;

  llvm::json::Object Json();
};

}  // namespace iree_prof::graph

#endif  // IREE_PROF_GRAPH_H_
