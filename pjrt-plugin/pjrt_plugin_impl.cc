// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "pjrt_plugin_impl.h"

#include <memory>
#include <string>

#include "iree/base/status.h"
#include "iree/hal/api.h"

namespace iree::pjrt {
namespace {

struct ApiInstance {
  ApiInstance(iree_hal_driver_t* driver, Logger& logger, ConfigVars config_vars)
      : driver_(driver),
        logger_(logger),
        config_vars_(std::move(config_vars)) {}
  ~ApiInstance() { iree_hal_driver_release(driver_); }

  static ApiInstance* Unwrap(PJRT_Api* api) {
    return static_cast<ApiInstance*>(api->priv);
  }

  iree_hal_driver_t* driver_;
  Logger logger_;
  ConfigVars config_vars_;
};

// ---------------------------------- Errors -----------------------------------

// Since PJRT errors can have their message queried, and it is assumed owned
// by the error, we always have to allocate/wrap :(
class ErrorInstance {
 public:
  ErrorInstance(iree_status_t status) : status_(status) {}
  ~ErrorInstance() { iree_status_ignore(status_); }
  static const ErrorInstance* FromError(const PJRT_Error* error) {
    return reinterpret_cast<const ErrorInstance*>(error);
  }

  iree_status_t status() const { return status_; }
  const std::string& message() const {
    if (cached_message_.empty()) {
      std::string buffer;
      iree_host_size_t actual_len;
      buffer.resize(128);
      if (!iree_status_format(status_, buffer.size(), buffer.data(),
                              &actual_len)) {
        buffer.resize(actual_len);
        if (!iree_status_format(status_, buffer.size(), buffer.data(),
                                &actual_len)) {
          actual_len = 0;
        }
      }
      buffer.resize(actual_len);
      cached_message_ = std::move(buffer);
    }
    return cached_message_;
  }

 private:
  iree_status_t status_;
  mutable std::string cached_message_;
};

PJRT_Error* MakeError(iree_status_t status) {
  if (iree_status_is_ok(status)) {
    return nullptr;
  }
  auto alloced_error = std::make_unique<ErrorInstance>(status);
  return reinterpret_cast<PJRT_Error*>(alloced_error.release());
}

void Error_Destroy_Impl(PJRT_Error_Destroy_Args* args) {
  if (!args->error) return;
  delete ErrorInstance::FromError(args->error);
}

void Error_Message_Impl(PJRT_Error_Message_Args* args) {
  auto* error = ErrorInstance::FromError(args->error);
  if (!error) {
    args->message = "OK";
    args->message_size = 2;
    return;
  }

  const std::string& message = error->message();
  args->message = message.data();
  args->message_size = message.size();
}

PJRT_Error* Error_GetCode_Impl(PJRT_Error_GetCode_Args* args) {
  auto* error = ErrorInstance::FromError(args->error);
  iree_status_code_t status_code = iree_status_code(error->status());
  switch (status_code) {
    case IREE_STATUS_CANCELLED:
      args->code = PJRT_Error_Code_CANCELLED;
    case IREE_STATUS_UNKNOWN:
      args->code = PJRT_Error_Code_UNKNOWN;
    case IREE_STATUS_INVALID_ARGUMENT:
      args->code = PJRT_Error_Code_INVALID_ARGUMENT;
    case IREE_STATUS_DEADLINE_EXCEEDED:
      args->code = PJRT_Error_Code_DEADLINE_EXCEEDED;
    case IREE_STATUS_NOT_FOUND:
      args->code = PJRT_Error_Code_NOT_FOUND;
    case IREE_STATUS_ALREADY_EXISTS:
      args->code = PJRT_Error_Code_ALREADY_EXISTS;
    case IREE_STATUS_PERMISSION_DENIED:
      args->code = PJRT_Error_Code_PERMISSION_DENIED;
    case IREE_STATUS_RESOURCE_EXHAUSTED:
      args->code = PJRT_Error_Code_RESOURCE_EXHAUSTED;
    case IREE_STATUS_FAILED_PRECONDITION:
      args->code = PJRT_Error_Code_FAILED_PRECONDITION;
    case IREE_STATUS_ABORTED:
      args->code = PJRT_Error_Code_ABORTED;
    case IREE_STATUS_OUT_OF_RANGE:
      args->code = PJRT_Error_Code_OUT_OF_RANGE;
    case IREE_STATUS_UNIMPLEMENTED:
      args->code = PJRT_Error_Code_UNIMPLEMENTED;
    case IREE_STATUS_INTERNAL:
      args->code = PJRT_Error_Code_INTERNAL;
    case IREE_STATUS_UNAVAILABLE:
      args->code = PJRT_Error_Code_UNAVAILABLE;
    case IREE_STATUS_DATA_LOSS:
      args->code = PJRT_Error_Code_DATA_LOSS;
    case IREE_STATUS_UNAUTHENTICATED:
      args->code = PJRT_Error_Code_UNAUTHENTICATED;
    case IREE_STATUS_DEFERRED:
      args->code = PJRT_Error_Code_UNKNOWN;  // No mapping
    default:
      // Should not happen.
      args->code = PJRT_Error_Code_UNKNOWN;
  }
  return nullptr;
}

// ---------------------------------- Client -----------------------------------

struct ClientInstance {
  ClientInstance(ApiInstance* api) : api(api) {}
  static ClientInstance* Unwrap(PJRT_Client* client) {
    return reinterpret_cast<ClientInstance*>(client);
  }
  ApiInstance* api;
};

PJRT_Error* Client_Create_Impl(PJRT_Client_Create_Args* args) {
  auto client =
      std::make_unique<ClientInstance>(ApiInstance::Unwrap(args->api));

  // Successful return.
  args->client = reinterpret_cast<PJRT_Client*>(client.release());
  return nullptr;
}

PJRT_Error* Client_Destroy_Impl(PJRT_Client_Destroy_Args* args) {
  delete ClientInstance::Unwrap(args->client);
  return nullptr;
}

}  // namespace

bool ConfigVars::Parse(Logger& logger, const char** config_vars,
                       size_t config_var_size) {
  // TODO
  return true;
}

bool Initialize(PJRT_Api* api, Logger& logger, ConfigVars config_vars) {
  api->struct_size = PJRT_Api_STRUCT_SIZE;

  // Populate functions.
  api->PJRT_Error_Destroy = Error_Destroy_Impl;
  api->PJRT_Error_Message = Error_Message_Impl;
  api->PJRT_Error_GetCode = Error_GetCode_Impl;

  api->PJRT_Client_Create = Client_Create_Impl;
  api->PJRT_Client_Destroy = Client_Destroy_Impl;

  // Attempt to create the requested driver.
  iree_hal_driver_t* driver;
  iree_status_t status = iree_hal_driver_registry_try_create(
      iree_hal_driver_registry_default(),
      iree_make_string_view(config_vars.driver_name.data(),
                            config_vars.driver_name.size()),
      iree_allocator_system(), &driver);
  if (!iree_status_is_ok(status)) {
    // Can't return an error in API initialization. Just log.
    ErrorInstance error(status);
    std::string message("Could not initialize IREE driver '");
    message.append(config_vars.driver_name);
    message.append("': ");
    message.append(error.message());
    logger.error(message);
    return false;
  }

  // Cannot fail after this: transfer ownership.
  auto api_instance =
      std::make_unique<ApiInstance>(driver, logger, std::move(config_vars));
  api->priv = api_instance.release();
  return true;
}

void Deinitialize(PJRT_Api* api) { delete ApiInstance::Unwrap(api); }

}  // namespace iree::pjrt
