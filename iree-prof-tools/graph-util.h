// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PROF_GRAPH_UTIL_H_
#define IREE_PROF_GRAPH_UTIL_H_

#include "iree-prof-tools/graph.h"
#include "third_party/abseil-cpp/absl/status/statusor.h"
#include "third_party/abseil-cpp/absl/strings/string_view.h"
#include "third_party/llvm-project/mlir/include/mlir/IR/BuiltinOps.h"

namespace iree_prof::graph {

// Builds a GraphCollection from |entrypoint| in |module| with |label|.
// If |entrypoint| is empty, finds a function with most stream ops.
absl::StatusOr<GraphCollection> GetGraphCollection(
    mlir::ModuleOp module,
    absl::string_view label,
    absl::string_view entrypoint,
    int min_num_ops_to_group);

}  // namespace iree_prof::graph

#endif  // IREE_PROF_GRAPH_UTIL_H_
