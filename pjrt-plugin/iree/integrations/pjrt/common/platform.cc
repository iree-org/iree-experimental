// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/integrations/pjrt/common/platform.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

namespace iree::pjrt {

//===----------------------------------------------------------------------===//
// ConfigVars
//===----------------------------------------------------------------------===//

void ConfigVars::EnableEnvFallback(std::string env_fallback_prefix) {
  env_fallback_prefix_ = env_fallback_prefix;
}

std::optional<std::string> ConfigVars::Lookup(const std::string& key) {
  auto found_it = kv_entries_.find(key);
  if (found_it != kv_entries_.end()) {
    return found_it->second;
  }

  // Env fallback?
  if (!env_fallback_prefix_) return {};

  std::string full_env_key = *env_fallback_prefix_;
  full_env_key.append(key);
  char* found_env = std::getenv(full_env_key.c_str());
  if (found_env) {
    return std::string(found_env);
  }

  return {};
}

//===----------------------------------------------------------------------===//
// Logger
//===----------------------------------------------------------------------===//

void Logger::debug(std::string_view message) {
  std::cerr << "[IREE-PJRT] DEBUG: " << message << std::endl;
}

void Logger::error(std::string_view message) {
  std::cerr << "[IREE-PJRT] ERROR: " << message << std::endl;
}

//===----------------------------------------------------------------------===//
// ArtifactDumper
//===----------------------------------------------------------------------===//

ArtifactDumper::~ArtifactDumper() = default;

std::unique_ptr<ArtifactDumper::Transaction>
ArtifactDumper::CreateTransaction() {
  return nullptr;
}

std::string ArtifactDumper::DebugString() { return std::string("disabled"); }

//===----------------------------------------------------------------------===//
// FilesArtifactDumper
//===----------------------------------------------------------------------===//

class FilesArtifactDumper::FilesTransaction
    : public ArtifactDumper::Transaction {
 public:
  FilesTransaction(Logger& logger, std::filesystem::path base_path,
                   int64_t transaction_id, bool retain_all)
      : logger_(logger),
        base_path_(std::move(base_path)),
        transaction_id_(transaction_id),
        retain_all_(retain_all) {}
  ~FilesTransaction() { Retain(); }

  void WriteArtifact(std::string_view label, std::string_view extension,
                     int index, std::string_view contents) override {
    std::string basename = std::to_string(transaction_id_);
    basename.append("-");
    basename.append(label);
    if (index >= 0) {
      basename.append(std::to_string(index));
    }
    basename.append(".");
    basename.append(extension);

    auto file_path = base_path_ / basename;
    std::ofstream fout;
    fout.open(file_path, std::ofstream::out | std::ofstream::trunc |
                             std::ofstream::binary);
    fout.write(contents.data(), contents.size());
    fout.close();

    written_paths_.push_back(file_path);

    if (!fout.good()) {
      std::string message("I/O error dumping artifact: ");
      message.append(file_path);
      logger_.error(message);
    }
  }

  void Retain() override {
    if (written_paths_.empty()) return;

    std::string message;
    message.append("Retained artifacts in: ");
    message.append(base_path_);
    for (auto& p : written_paths_) {
      message.append("\n  ");
      message.append(p.filename());
    }
    logger_.debug(message);

    written_paths_.clear();
  }

  void Cancel() override {
    if (retain_all_) {
      Retain();
      return;
    }

    for (auto& p : written_paths_) {
      std::error_code ec;
      std::filesystem::remove(p, ec);
      if (ec) {
        // Only carp as a debug message since there are legitimate reasons
        // this can happen depending on what is going on at the system level.
        std::string message("Error removing artifact: ");
        message.append(p);
        logger_.debug(message);
      }
    }

    written_paths_.clear();
  }

 private:
  Logger& logger_;
  std::vector<std::filesystem::path> written_paths_;
  std::filesystem::path base_path_;
  int64_t transaction_id_;
  bool retain_all_;
};

FilesArtifactDumper::FilesArtifactDumper(Logger& logger,
                                         std::string_view path_spec,
                                         bool retain_all)
    : logger_(logger), retain_all_(retain_all) {
  path_ = path_spec;
  enabled_ = true;

  std::error_code ec;
  std::filesystem::create_directories(path_, ec);
  if (ec) {
    std::string message("Error creating artifact directory '");
    message.append(path_);
    message.append("' (artifact dumping disabled): ");
    message.append(ec.message());
    logger_.error(message);
    enabled_ = false;
  }
}

FilesArtifactDumper::~FilesArtifactDumper() = default;

std::unique_ptr<ArtifactDumper::Transaction>
FilesArtifactDumper::CreateTransaction() {
  return std::make_unique<FilesTransaction>(
      logger_, path_, next_transaction_id_.fetch_add(1), retain_all_);
}

std::string FilesArtifactDumper::DebugString() { return path_; }

//===----------------------------------------------------------------------===//
// Platform
//===----------------------------------------------------------------------===//

Platform::~Platform() = default;

iree_status_t Platform::Initialize() {
  IREE_RETURN_IF_ERROR(SubclassInitialize());

  if (!logger_ || !compiler_ || !artifact_dumper_) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "the Platform failed to initialize all objects");
  }
  return iree_ok_status();
}

}  // namespace iree::pjrt
