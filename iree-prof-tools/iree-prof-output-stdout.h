// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PROF_OUTPUT_STDOUT_H_
#define IREE_PROF_OUTPUT_STDOUT_H_

#include <string>
#include <vector>

#include "iree-prof-tools/iree-prof-output.h"

namespace iree_prof {

// Output IREE profiling results in a tracy worker to stdout.
class IreeProfOutputStdout : public IreeProfOutput {
 public:
  enum class DurationUnit {
    kNotSpecified,
    kNanoseconds,
    kMicroseconds,
    kMilliseconds,
    kSeconds,
  };

  IreeProfOutputStdout(const std::vector<std::string>& zone_substrs,
                       const std::vector<std::string>& thread_substrs,
                       DurationUnit unit);
  ~IreeProfOutputStdout() override;

  IreeProfOutputStdout(const IreeProfOutputStdout&) = delete;
  IreeProfOutputStdout& operator=(const IreeProfOutputStdout&) = delete;

  // IreeProfOutput implementation:
  absl::Status Output(tracy::Worker& worker) override;

 private:
  const std::vector<std::string> zone_substrs_;
  const std::vector<std::string> thread_substrs_;
  const DurationUnit unit_;
};

}  // namespace iree_prof

#endif  // IREE_PROF_OUTPUT_STDOUT_H_
