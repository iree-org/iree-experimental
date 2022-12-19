// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_COMMON_API_IMPL_H_
#define IREE_PJRT_PLUGIN_PJRT_COMMON_API_IMPL_H_

#include <string>
#include <string_view>

#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"

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
};

//===----------------------------------------------------------------------===//
// Globals
// Static globals passed to API objects on construction.
//===----------------------------------------------------------------------===//

class Globals {
 public:
  Globals(Logger& logger, ConfigVars config_vars)
      : logger_(logger), config_vars_(std::move(config_vars)) {}

  Logger& logger() { return logger_; }

 private:
  Logger& logger_;
  ConfigVars config_vars_;
};

//===----------------------------------------------------------------------===//
// PJRT_Error wrapper
// PJRT Errors are simple wrappers around an iree_status_t. They are
// infrequently created, so we make some ergonomic concessions (caching
// messages, etc).
//===----------------------------------------------------------------------===//

class ErrorInstance {
 public:
  ErrorInstance(iree_status_t status) : status_(status) {}
  ~ErrorInstance() { iree_status_ignore(status_); }
  static void BindApi(PJRT_Api* api);

  static const ErrorInstance* FromError(const PJRT_Error* error) {
    return reinterpret_cast<const ErrorInstance*>(error);
  }

  iree_status_t status() const { return status_; }
  const std::string& message() const;

 private:
  iree_status_t status_;
  mutable std::string cached_message_;
};

inline PJRT_Error* MakeError(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return nullptr;
  }
  auto alloced_error = std::make_unique<ErrorInstance>(status);
  return reinterpret_cast<PJRT_Error*>(alloced_error.release());
}

//===----------------------------------------------------------------------===//
// ClientInstance
// The root of the runtime hierarchy, these map to an IREE driver and are
// created against an API.
//===----------------------------------------------------------------------===//

struct ClientInstance {
 public:
  ClientInstance(Globals& globals) : globals_(globals) {}
  virtual ~ClientInstance();

  // Binds monomorphic entry-points for the client.
  static void BindApi(PJRT_Api* api);

  static ClientInstance* Unwrap(PJRT_Client* client) {
    return reinterpret_cast<ClientInstance*>(client);
  }

  // Before the client is usable, it must be initialized.
  PJRT_Error* Initialize();

  // Must be defined by concrete subclasses.
  virtual PJRT_Error* CreateDriver(iree_hal_driver_t** out_driver) = 0;

  Globals& globals() { return globals_; }
  Logger& logger() { return globals_.logger(); }

 private:
  iree_hal_driver_t* driver_ = nullptr;
  Globals& globals_;
};

// Binds all monomorphic API members and top-level API struct setup.
void BindMonomorphicApi(PJRT_Api* api, Globals& globals);

// Fully binds the PJRT_Api struct for all types. Polymorphic types must be
// specified by template parameters.
template <typename ClientInstanceTy>
static void BindApi(PJRT_Api* api) {
  // TODO: We should be stashing Globals* on api->priv and then constructor
  // function arguments should have a member pointing back to the api so they
  // can retrieve it. Once that is done, don't hardcode here.
  static Logger logger;
  static Globals globals(logger, ConfigVars());

  BindMonomorphicApi(api, globals);

  // Bind polymorphic entry-points.
  api->PJRT_Client_Create = +[](PJRT_Client_Create_Args* args) -> PJRT_Error* {
    auto client = std::make_unique<ClientInstanceTy>(globals);
    auto* error = client->Initialize();
    if (error) return error;

    // Successful return.
    args->client = reinterpret_cast<PJRT_Client*>(client.release());
    return nullptr;
  };
}

}  // namespace iree::pjrt

#endif  // IREE_PJRT_PLUGIN_PJRT_COMMON_API_IMPL_H_
