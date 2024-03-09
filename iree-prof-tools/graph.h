// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PROF_GRAPH_H_
#define IREE_PROF_GRAPH_H_

#include <memory>
#include <optional>
#include <string>

#include "third_party/abseil-cpp/absl/strings/str_cat.h"
#include "third_party/llvm-project/llvm/include/llvm/Support/JSON.h"

namespace iree_prof::graph {

// An object with a pair of |key| and |value|.
struct Attribute {
  Attribute(std::string key, std::string value)
      : key(std::move(key)), value(std::move(value)) {}
  Attribute(std::string key, int64_t int_value)
      : key(std::move(key)), value(absl::StrCat(int_value)),
        int_value(int_value) {}

  std::string key;
  std::string value;
  std::optional<int64_t> int_value;

  llvm::json::Object Json() const;
};

// An item in output metadata of a node.
struct Metadata {
  std::string id;
  std::vector<Attribute> attrs;

  llvm::json::Object Json() const;
};

// An incoming edge in the graph.
struct IncomingEdge {
  // The id of the source node (where the edge comes from).
  std::string source_node_id;
  // The id of the output from the source node that this edge goes out of.
  std::string source_node_output_id;
  // The id of the input from the target node (this node) that this edge
  // connects to.
  std::string target_node_input_id;
  // Other associated metadata for this edge.
  std::vector<Attribute> metadata;

  llvm::json::Object Json() const;
};

// A single node in the graph.
struct GraphNode {
  // The unique id of the node.
  std::string id;
  // The label of the node, displayed on the node in the model graph.
  std::string label;

  // The namespace/hierarchy data of the node in the form of a "path" (e.g.
  // a/b/c). Don't include the node label as the last component of the
  // namespace. The visualizer will use this data to visualize nodes in a nested
  // way.
  //
  // For example, for three nodes with the following label and namespace data:
  // - N1: a/b
  // - N2: a/b
  // - N3: a
  //
  // The visualizer will first show a collapsed box labeled 'a'. After the box
  // is expanded (by user clicking on it), it will show node N3, and another
  // collapsed box labeled 'b'. After the box 'b' is expanded, it will show two
  // nodes N1 and N2 inside the box 'b'.
  std::string node_namespace;

  // Ids of subgraphs that this node goes into.
  //
  // The graphs referenced here should be the ones from the |graphs| field in
  // GraphCollection below. Once set, users will be able to click this node,
  // pick a subgraph from a drop-down list, and see the visualization for the
  // selected subgraph.
  std::vector<std::string> subgraph_ids;

  // The attributes of the node.
  std::vector<Attribute> node_attrs;
  // A list of incoming edges.
  std::vector<IncomingEdge> incoming_edges;
  // Metadata for outputs.
  std::vector<Metadata> outputs_metadata;

  llvm::json::Object Json() const;
};

// A graph to be visualized. It is passed into the visualizer through
// GraphCollection below.*
//
// Graphs can also be `linked` together through the `subgraphIds` field of a
// node.
struct Graph {
  explicit Graph(std::string id) : id(std::move(id)) {}

  // The id of the graph.
  std::string id;
  // A list of nodes in the graph.
  // Differently from others, it uses unique_ptr to make references valid after
  // adding more nodes.
  std::vector<std::unique_ptr<GraphNode>> nodes;

  llvm::json::Object Json() const;
};

// A collection of graphs. This is the input to the visualizer.
// The visualizer accepts a list of graphs. The first graph in the list is the
// default one to be visualized. Users can pick a different graph from a
// drop-down list.
struct GraphCollection {
  // The label of the collection.
  std::string label;
  // The graphs inside the collection.
  std::vector<Graph> graphs;

  llvm::json::Object Json() const;
};

}  // namespace iree_prof::graph

#endif  // IREE_PROF_GRAPH_H_
