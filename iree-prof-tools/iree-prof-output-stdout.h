// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PROF_OUTPUT_STDOUT_H_
#define IREE_PROF_OUTPUT_STDOUT_H_

#include <regex>
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

  IreeProfOutputStdout(bool output_zone_stats,
                       bool output_per_op_stats,
                       const std::string& zone_regex,
                       const std::string& thread_regex,
                       DurationUnit unit);
  ~IreeProfOutputStdout() override;

  IreeProfOutputStdout(const IreeProfOutputStdout&) = delete;
  IreeProfOutputStdout& operator=(const IreeProfOutputStdout&) = delete;

  // IreeProfOutput implementation:
  absl::Status Output(tracy::Worker& worker) override;

 private:
  const bool output_zone_stats_;
  const bool output_per_op_stats_;
  const std::regex zone_regex_;
  const std::regex thread_regex_;
  const DurationUnit unit_;
};

}  // namespace iree_prof

#endif  // IREE_PROF_OUTPUT_STDOUT_H_
