// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-prof-tools/iree-prof-output.h"

#include <string>
#include <vector>

#include "iree-prof-tools/iree-prof-output-chrome.h"
#include "iree-prof-tools/iree-prof-output-stdout.h"
#include "iree-prof-tools/iree-prof-output-tracy.h"
#include "third_party/abseil-cpp/absl/flags/flag.h"
#include "third_party/abseil-cpp/absl/log/log.h"
#include "third_party/abseil-cpp/absl/status/status.h"
#include "third_party/tracy/server/TracyWorker.hpp"

ABSL_FLAG(std::string, output_tracy_file, "",
          "Tracy file to write as the output of the given executable command.");
ABSL_FLAG(std::string, output_chrome_file, "",
          "Chrome tracing viewer json file to write as the output of execution "
          "or conversion.");
ABSL_FLAG(bool, output_stdout, true,
          "Whether to print Tracy result to stdout.");
ABSL_FLAG(std::vector<std::string>, zone_substrs,
          std::vector<std::string>({"iree_hal_buffer_map_", "_dispatch_"}),
          "Comma-separated substrings of tracy zones to output to stdout. "
          "If empty, no zones will be output.");
ABSL_FLAG(std::vector<std::string>, thread_substrs, {},
          "Comma-separated substrings of threads to output to stdout. "
          "If empty, all thread including main threads will be output.");
ABSL_FLAG(std::string, duration_unit, "milliseconds",
          "Unit of duration of zone to output to stdout. It must be one of "
          "seconds(s), millseconds(ms), microseconds(us), or nanoseconds(ns).");

namespace iree_prof {
namespace {

void LogStatusIfError(const absl::Status& status) {
  if (!status.ok()) {
    LOG(ERROR) << status;
  }
}

IreeProfOutputStdout::DurationUnit ToUnit(const absl::string_view flag) {
  if (flag == "seconds" || flag == "s") {
    return IreeProfOutputStdout::DurationUnit::kSeconds;
  }
  if (flag == "milliseconds" || flag == "ms") {
    return IreeProfOutputStdout::DurationUnit::kMilliseconds;
  }
  if (flag == "microseconds" || flag == "us") {
    return IreeProfOutputStdout::DurationUnit::kMicroseconds;
  }
  if (flag == "nanoseconds" || flag == "ns") {
    return IreeProfOutputStdout::DurationUnit::kNanoseconds;
  }
  return IreeProfOutputStdout::DurationUnit::kNotSpecified;
}

}  // namespace

void Output(tracy::Worker& worker) {
  if (absl::GetFlag(FLAGS_output_stdout)) {
    LogStatusIfError(
        IreeProfOutputStdout(absl::GetFlag(FLAGS_zone_substrs),
                             absl::GetFlag(FLAGS_thread_substrs),
                             ToUnit(absl::GetFlag(FLAGS_duration_unit)))
        .Output(worker));
  }

  std::string output_tracy_file = absl::GetFlag(FLAGS_output_tracy_file);
  if (!output_tracy_file.empty()) {
    LogStatusIfError(IreeProfOutputTracy(output_tracy_file).Output(worker));
  }

  std::string output_chrome_file = absl::GetFlag(FLAGS_output_chrome_file);
  if (!output_chrome_file.empty()) {
    LogStatusIfError(IreeProfOutputChrome(output_chrome_file).Output(worker));
  }
}

}  // namespace iree_prof
