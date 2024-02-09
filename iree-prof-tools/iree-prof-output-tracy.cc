// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-prof-tools/iree-prof-output-tracy.h"

#include <memory>

#include "third_party/abseil-cpp/absl/status/status.h"
#include "third_party/abseil-cpp/absl/strings/str_cat.h"
#include "third_party/abseil-cpp/absl/strings/string_view.h"
#include "third_party/tracy/server/TracyFileWrite.hpp"
#include "third_party/tracy/server/TracyWorker.hpp"

namespace iree_prof {

IreeProfOutputTracy::IreeProfOutputTracy(absl::string_view output_file_path)
    : output_file_path_(output_file_path) {}

IreeProfOutputTracy::~IreeProfOutputTracy() = default;

absl::Status IreeProfOutputTracy::Output(tracy::Worker& worker) {
  auto f = std::unique_ptr<tracy::FileWrite>(
      tracy::FileWrite::Open(output_file_path_.c_str()));
  if (!f) {
    return absl::UnavailableError(
        absl::StrCat("Could not write tracy file ", output_file_path_));
  }

  worker.Write(*f, /*fiDict=*/false);
  return absl::OkStatus();
}

}  // namespace iree_prof
