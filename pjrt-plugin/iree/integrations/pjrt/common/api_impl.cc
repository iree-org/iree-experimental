// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/integrations/pjrt/common/api_impl.h"

#include <iostream>

namespace iree::pjrt {

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
// Error
//===----------------------------------------------------------------------===//

void ErrorInstance::BindApi(PJRT_Api* api) {
  api->PJRT_Error_Destroy = +[](PJRT_Error_Destroy_Args* args) {
    if (!args->error) return;
    delete ErrorInstance::FromError(args->error);
  };
  api->PJRT_Error_Message = +[](PJRT_Error_Message_Args* args) {
    auto* error = ErrorInstance::FromError(args->error);
    if (!error) {
      args->message = "OK";
      args->message_size = 2;
      return;
    }

    const std::string& message = error->message();
    args->message = message.data();
    args->message_size = message.size();
  };
  api->PJRT_Error_GetCode = +[](PJRT_Error_GetCode_Args* args) -> PJRT_Error* {
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
  };
}

const std::string& ErrorInstance::message() const {
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

//===----------------------------------------------------------------------===//
// ClientInstance
//===----------------------------------------------------------------------===//

ClientInstance::~ClientInstance() = default;

void ClientInstance::BindApi(PJRT_Api* api) {
  api->PJRT_Client_Destroy =
      +[](PJRT_Client_Destroy_Args* args) -> PJRT_Error* {
    delete ClientInstance::Unwrap(args->client);
    return nullptr;
  };
}

PJRT_Error* ClientInstance::Initialize() {
  auto* error = CreateDriver(&driver_);
  if (error) return error;

  // More initialization.
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Top-level API binding.
//===----------------------------------------------------------------------===//

void BindMonomorphicApi(PJRT_Api* api, Globals& globals) {
  api->struct_size = PJRT_Api_STRUCT_SIZE;
  api->priv = &globals;

  // Bind by object types.
  ClientInstance::BindApi(api);
  ErrorInstance::BindApi(api);
}

}  // namespace iree::pjrt
