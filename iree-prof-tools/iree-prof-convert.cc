// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <string>
#include <vector>

#include "iree-prof-tools/iree-prof-output.h"
#include "iree-prof-tools/iree-prof-output-utils.h"
#include "third_party/abseil-cpp/absl/flags/flag.h"
#include "third_party/abseil-cpp/absl/log/log.h"
#include "third_party/tracy/server/TracyFileRead.hpp"
#include "third_party/tracy/server/TracyWorker.hpp"

ABSL_FLAG(std::string, input_tracy_file, "",
          "Tracy file to read. Ignored if executable command is given.");

int main(int argc, char** argv) {
  std::vector<char*> remain_args =
      iree_prof::InitializeLogAndParseCommandLine(argc, argv);
  if (remain_args.empty()) {
    LOG(WARNING) << remain_args.size() << " unknown flags exist.";
  }

  std::string input_tracy_file = absl::GetFlag(FLAGS_input_tracy_file);
  if (input_tracy_file.empty()) {
    LOG(ERROR) << "Tracy file to read is not provided.";
    return EXIT_FAILURE;
  }

  auto f = std::unique_ptr<tracy::FileRead>(
      tracy::FileRead::Open(input_tracy_file.c_str()));
  if (!f) {
    LOG(ERROR) << "Could not open file: " << input_tracy_file;
    return EXIT_FAILURE;
  }

  auto worker = std::make_unique<tracy::Worker>(*f);
  while (!worker->AreSourceLocationZonesReady()) {
    iree_prof::YieldCpu();
  }

  iree_prof::Output(*worker);
  return EXIT_SUCCESS;
}
