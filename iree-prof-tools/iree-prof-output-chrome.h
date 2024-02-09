// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PROF_OUTPUT_CHROME_H_
#define IREE_PROF_OUTPUT_CHROME_H_

#include <string>

#include "iree-prof-tools/iree-prof-output.h"
#include "third_party/abseil-cpp/absl/strings/string_view.h"

namespace iree_prof {

// Output IREE profiling results in a tracy worker to a JSON file which can be
// loaded in the chrome tracing viewer, chrome://tracing or perfetto UI,
// https://ui.perfetto.dev/.
class IreeProfOutputChrome : public IreeProfOutput {
 public:
  explicit IreeProfOutputChrome(absl::string_view output_file_path);
  ~IreeProfOutputChrome() override;

  IreeProfOutputChrome(const IreeProfOutputChrome&) = delete;
  IreeProfOutputChrome& operator=(const IreeProfOutputChrome&) = delete;

  // IreeProfOutput implementation:
  absl::Status Output(tracy::Worker& worker) override;

 private:
  std::string output_file_path_;
};

}  // namespace iree_prof

#endif  // IREE_PROF_OUTPUT_CHROME_H_
