// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PROF_OUTPUT_H_
#define IREE_PROF_OUTPUT_H_

#include "third_party/abseil-cpp/absl/status/status.h"
#include "third_party/tracy/server/TracyWorker.hpp"

namespace iree_prof {

// Abstract interface for various outputs of IREE profiling results in a tracy
// worker.
class IreeProfOutput {
 public:
  virtual ~IreeProfOutput() = default;

  // Outputs tracy wroker information in a specific format the subclass
  // implemented.
  virtual absl::Status Output(tracy::Worker& worker) = 0;
};

// Output IREE profiling results in various ways according to the command line
// flags.
void Output(tracy::Worker& worker);

}  // namespace iree_prof

#endif  // IREE_PROF_OUTPUT_H_
