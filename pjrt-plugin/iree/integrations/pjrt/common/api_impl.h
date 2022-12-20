// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_PJRT_PLUGIN_PJRT_COMMON_API_IMPL_H_
#define IREE_PJRT_PLUGIN_PJRT_COMMON_API_IMPL_H_

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "iree/base/status.h"
#include "iree/hal/api.h"
#include "iree/integrations/pjrt/common/compiler.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"

namespace iree::pjrt {

class ClientInstance;
class ConfigVars;
class DeviceInstance;
class ErrorInstance;
class Globals;
class Logger;

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
// BufferInstance
//===----------------------------------------------------------------------===//

class BufferInstance {
 public:
  BufferInstance(ClientInstance& client) : client_(client) {}
  operator PJRT_Buffer*() { return reinterpret_cast<PJRT_Buffer*>(this); }
  static BufferInstance* Unwrap(PJRT_Buffer* buffer) {
    return reinterpret_cast<BufferInstance*>(buffer);
  }
  static void BindApi(PJRT_Api* api);

 private:
  ClientInstance& client_;
};

//===----------------------------------------------------------------------===//
// DeviceInstance
//===----------------------------------------------------------------------===//

class DeviceInstance {
 public:
  DeviceInstance(int client_id, ClientInstance& client,
                 iree_hal_device_info_t* info)
      : client_id_(client_id), client_(client), info_(info) {}
  operator PJRT_Device*() { return reinterpret_cast<PJRT_Device*>(this); }
  static void BindApi(PJRT_Api* api);
  static DeviceInstance* Unwrap(PJRT_Device* device) {
    return reinterpret_cast<DeviceInstance*>(device);
  }

  // Since the PJRT device id is a simple int and the IREE device_id is
  // a pointer-sized value, we just assign a synthetic id. Currently, this
  // is the offset into the devices() array on the client. Will need to be
  // revisited if ever supporting re-scanning (but many things would seem to
  // need updates then).
  int client_id() { return client_id_; }
  iree_hal_device_info_t* info() { return info_; }

  // Not yet implemented but plumbed through.
  bool is_addressable() { return true; }
  int process_index() { return 0; }

 private:
  int client_id_;
  ClientInstance& client_;
  iree_hal_device_info_t* info_;
};

//===----------------------------------------------------------------------===//
// EventInstance
//===----------------------------------------------------------------------===//

class EventInstance {
 public:
  EventInstance() = default;
  operator PJRT_Event*() { return reinterpret_cast<PJRT_Event*>(this); }
  static void BindApi(PJRT_Api* api);
  static EventInstance* Unwrap(PJRT_Event* exe) {
    return reinterpret_cast<EventInstance*>(exe);
  }
};

//===----------------------------------------------------------------------===//
// ExecutableInstance
//===----------------------------------------------------------------------===//

class ExecutableInstance {
 public:
  ExecutableInstance(ClientInstance& client,
                     std::unique_ptr<CompilerOutput> binary,
                     const std::vector<DeviceInstance*>& addressable_devices)
      : client_(client),
        binary_(std::move(binary)),
        addressable_devices_(addressable_devices) {}
  operator PJRT_Executable*() {
    return reinterpret_cast<PJRT_Executable*>(this);
  }
  static void BindApi(PJRT_Api* api);
  static ExecutableInstance* Unwrap(PJRT_Executable* exe) {
    return reinterpret_cast<ExecutableInstance*>(exe);
  }

  const std::vector<DeviceInstance*>& addressable_devices() {
    return addressable_devices_;
  }

 private:
  ClientInstance& client_;
  std::unique_ptr<CompilerOutput> binary_;
  std::vector<DeviceInstance*> addressable_devices_;
};

//===----------------------------------------------------------------------===//
// ClientInstance
// The root of the runtime hierarchy, these map to an IREE driver and are
// created against an API.
//===----------------------------------------------------------------------===//

struct ClientInstance {
 public:
  ClientInstance(Globals& globals);
  virtual ~ClientInstance();

  // Binds monomorphic entry-points for the client.
  static void BindApi(PJRT_Api* api);

  static ClientInstance* Unwrap(PJRT_Client* client) {
    return reinterpret_cast<ClientInstance*>(client);
  }

  // Before the client is usable, it must be initialized.
  PJRT_Error* Initialize();

  // Must be defined by concrete subclasses.
  virtual iree_status_t CreateDriver(iree_hal_driver_t** out_driver) = 0;

  Globals& globals() { return globals_; }
  Logger& logger() { return globals_.logger(); }

  const std::vector<DeviceInstance*>& devices() { return devices_; }
  const std::vector<DeviceInstance*>& addressable_devices() {
    return addressable_devices_;
  }
  const std::string& cached_platform_name() { return cached_platform_name_; }
  const std::string& cached_platform_version() {
    return cached_platform_version_;
  }

  // Compiles.
  // See TODOs in PJRT_Client_Compile.
  PJRT_Error* Compile(PJRT_Program* program, ExecutableInstance** executable);

 protected:
  iree_allocator_t host_allocator_;
  std::string cached_platform_name_;
  std::string cached_platform_version_;

 private:
  iree_status_t InitializeCompiler();
  iree_status_t PopulateDevices();

  // Populated during initialization.
  std::shared_ptr<AbstractCompiler> compiler_;
  iree_hal_driver_t* driver_ = nullptr;
  iree_hal_device_info_t* device_infos_ = nullptr;
  iree_host_size_t device_info_count_ = 0;
  std::vector<DeviceInstance*> devices_;
  std::vector<DeviceInstance*> addressable_devices_;
  Globals& globals_;
};

//===----------------------------------------------------------------------===//
// API binding
//===----------------------------------------------------------------------===//

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