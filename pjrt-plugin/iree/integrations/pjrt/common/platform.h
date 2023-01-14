// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_PLATFORM_H_
#define IREE_PJRT_PLUGIN_PJRT_PLATFORM_H_

#include <atomic>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

#include "iree/base/status.h"
#include "iree/integrations/pjrt/common/compiler.h"

namespace iree::pjrt {

//===----------------------------------------------------------------------===//
// Logger
// The plugin API currently does not have any logging facilities, but since
// these are easier added later, we have a placeholder Logger that we thread
// through. It can be extended later.
//===----------------------------------------------------------------------===//

class Logger {
 public:
  Logger() = default;
  void debug(std::string_view message);
  void error(std::string_view message);
};

//===----------------------------------------------------------------------===//
// ConfigVars
// Placeholder for API-level configuration (i.e. from environment, files, etc).
//===----------------------------------------------------------------------===//

struct ConfigVars {
 public:
  ConfigVars() = default;
  // Configures this instance to fallback to the system environment if a config
  // var is not found by prefixing |env_fallback_prefix| to the variable name.
  void EnableEnvFallback(std::string env_fallback_prefix);

  // Looks up a variable value by name. On success, the resulting string_view
  // is valid at least until the next mutation.
  std::optional<std::string> Lookup(const std::string& key);

 private:
  std::unordered_map<std::string, std::string> kv_entries_;
  std::optional<std::string> env_fallback_prefix_;
};

//===----------------------------------------------------------------------===//
// ArtifactDumper
// The typical modes of operation are:
//   - Completely disabled
//   - Enabled to persist all artifacts
//   - Enabled to persist artifacts on recoverable errors (i.e. compile errors,
//     etc)
//   - Enabled to persist artifacts on crash only
// Since we often don't know what the outcome of a transaction will be right
// away, the transaction can be held open for some period of time and cancelled
// later.
//
// In general, since this is a debugging facility, it swallows any errors,
// reporting them to the Logger.
//
// The default implementation can be instantiated and is hard-coded to
// !enabled().
//===----------------------------------------------------------------------===//

class ArtifactDumper {
 public:
  class Transaction {
   public:
    virtual ~Transaction() = default;

    // Writes an artifact with a transaction-unique label/index.
    virtual void WriteArtifact(std::string_view label,
                               std::string_view extension, int index,
                               std::string_view contents) = 0;

    // Completes the transation and retains all artifacts.
    virtual void Retain() = 0;

    // Completes the transaction and removes all artifacts.
    virtual void Cancel() = 0;
  };

  virtual ~ArtifactDumper();

  // Not virtual for quick checks in disabled state.
  bool enabled() { return enabled_; }

  // Allocates a new transaction which can be used to append related
  // artifacts. All artifacts associated with a transaction can be removed upon
  // a successful transaction, or they will be retained upon crash or
  // recoverable error.
  // Must only be called if enabled().
  virtual std::unique_ptr<Transaction> CreateTransaction();

  // Returns a string suitable for emitting to the debug log, describing
  // where and how artifacts will be retained.
  virtual std::string DebugString();

 protected:
  bool enabled_ = false;
};

// Dumps artifacts to a path on the file system.
// This is initialized with a path_spec. Currently, this is just a path on
// the file system, but it should be brought into alignment with how the Python
// side does it.
// See:
// https://github.com/iree-org/iree/blob/main/compiler/src/iree/compiler/API/python/iree/compiler/tools/debugging.py#L29
class FilesArtifactDumper : public ArtifactDumper {
 public:
  class FilesTransaction;

  FilesArtifactDumper(Logger& logger, std::string_view path_spec,
                      bool retain_all);
  ~FilesArtifactDumper() override;

  std::unique_ptr<Transaction> CreateTransaction() override;
  std::string DebugString() override;

 private:
  Logger& logger_;
  std::atomic<int64_t> next_transaction_id_{0};
  std::string path_;
  bool retain_all_;
};

//===----------------------------------------------------------------------===//
// Platform
// Encapsulates aspects of the platform which may differ under different
// usage and/or environments.
//===----------------------------------------------------------------------===//

class Platform {
 public:
  virtual ~Platform();
  iree_status_t Initialize();
  ConfigVars& config_vars() { return config_vars_; }
  Logger& logger() { return *logger_; }
  AbstractCompiler& compiler() { return *compiler_; }
  ArtifactDumper& artifact_dumper() { return *artifact_dumper_; }

 protected:
  virtual iree_status_t SubclassInitialize() = 0;
  ConfigVars config_vars_;
  std::unique_ptr<Logger> logger_;
  std::unique_ptr<AbstractCompiler> compiler_;
  std::unique_ptr<ArtifactDumper> artifact_dumper_;
};

}  // namespace iree::pjrt

#endif  // IREE_PJRT_PLUGIN_PJRT_PLATFORM_H_
