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
// DeviceInstance
//===----------------------------------------------------------------------===//

void DeviceInstance::BindApi(PJRT_Api* api) {
  api->PJRT_Device_Id = +[](PJRT_Device_Id_Args* args) -> PJRT_Error* {
    args->id = DeviceInstance::Unwrap(args->device)->client_id();
    return nullptr;
  };
  api->PJRT_Device_ProcessIndex =
      +[](PJRT_Device_ProcessIndex_Args* args) -> PJRT_Error* {
    args->process_index = DeviceInstance::Unwrap(args->device)->process_index();
    return nullptr;
  };
  api->PJRT_Device_IsAddressable =
      +[](PJRT_Device_IsAddressable_Args* args) -> PJRT_Error* {
    args->is_addressable =
        DeviceInstance::Unwrap(args->device)->is_addressable();
    return nullptr;
  };

  api->PJRT_Device_Attributes =
      +[](PJRT_Device_Attributes_Args* args) -> PJRT_Error* {
    // TODO: Implement something.
    args->num_attributes = 0;
    args->attributes = nullptr;
    return nullptr;
  };
  api->PJRT_Device_Kind = nullptr;
  api->PJRT_Device_LocalHardwareId = nullptr;
  api->PJRT_Device_DebugString = nullptr;
  api->PJRT_Device_ToString = nullptr;
}

//===----------------------------------------------------------------------===//
// ClientInstance
//===----------------------------------------------------------------------===//

ClientInstance::ClientInstance(Globals& globals) : globals_(globals) {
  host_allocator_ = iree_allocator_system();
  cached_platform_version_ = "git";  // TODO: Plumb through version info.
}

ClientInstance::~ClientInstance() {
  for (auto* device : devices_) {
    delete device;
  }
  if (device_infos_) {
    iree_allocator_free(host_allocator_, device_infos_);
  }
}

void ClientInstance::BindApi(PJRT_Api* api) {
  // PJRT_Client_Create is polymorphic
  api->PJRT_Client_Destroy =
      +[](PJRT_Client_Destroy_Args* args) -> PJRT_Error* {
    delete ClientInstance::Unwrap(args->client);
    return nullptr;
  };
  api->PJRT_Client_PlatformName =
      +[](PJRT_Client_PlatformName_Args* args) -> PJRT_Error* {
    auto* client = ClientInstance::Unwrap(args->client);
    args->platform_name = client->cached_platform_name().data();
    args->platform_name_size = client->cached_platform_name().size();
    return nullptr;
  };
  api->PJRT_Client_ProcessIndex =
      +[](PJRT_Client_ProcessIndex_Args* args) -> PJRT_Error* {
    args->process_index = 0;
    return nullptr;
  };
  api->PJRT_Client_PlatformVersion =
      +[](PJRT_Client_PlatformVersion_Args* args) -> PJRT_Error* {
    auto* client = ClientInstance::Unwrap(args->client);
    args->platform_version = client->cached_platform_version().data();
    args->platform_version_size = client->cached_platform_version().size();
    return nullptr;
  };
  api->PJRT_Client_Devices =
      +[](PJRT_Client_Devices_Args* args) -> PJRT_Error* {
    auto& devices = ClientInstance::Unwrap(args->client)->devices();
    args->devices = reinterpret_cast<PJRT_Device**>(devices.data());
    args->num_devices = devices.size();
    return nullptr;
  };
  api->PJRT_Client_AddressableDevices =
      +[](PJRT_Client_AddressableDevices_Args* args) -> PJRT_Error* {
    auto& devices = ClientInstance::Unwrap(args->client)->devices();
    args->addressable_devices = reinterpret_cast<PJRT_Device**>(devices.data());
    args->num_addressable_devices = devices.size();
    return nullptr;
  };
  api->PJRT_Client_LookupDevice =
      +[](PJRT_Client_LookupDevice_Args* args) -> PJRT_Error* {
    auto& devices = ClientInstance::Unwrap(args->client)->devices();
    size_t id_as_size = args->id;
    if (id_as_size >= devices.size()) {
      return MakeError(
          iree_make_status(IREE_STATUS_OUT_OF_RANGE,
                           "because device id %d is invalid (%d devices known)",
                           (int)id_as_size, (int)devices.size()));
    }
    args->device = *devices[id_as_size];
    return nullptr;
  };
  api->PJRT_Client_Compile = nullptr;
  api->PJRT_Client_DefaultDeviceAssignment =
      +[](PJRT_Client_DefaultDeviceAssignment_Args* args) -> PJRT_Error* {
    // TODO: Something sensible.
    for (size_t i = 0; i < args->default_assignment_size; ++i) {
      args->default_assignment[i] = 0;
    }
    return nullptr;
  };
  api->PJRT_Client_BufferFromHostBuffer = nullptr;
}

PJRT_Error* ClientInstance::Initialize() {
  auto status = CreateDriver(&driver_);
  if (!iree_status_is_ok(status)) return MakeError(status);

  status = PopulateDevices();
  if (!iree_status_is_ok(status)) return MakeError(status);

  // More initialization.
  return nullptr;
}

iree_status_t ClientInstance::PopulateDevices() {
  IREE_RETURN_IF_ERROR(iree_hal_driver_query_available_devices(
      driver_, host_allocator_, &device_info_count_, &device_infos_));
  devices_.resize(device_info_count_);
  for (iree_host_size_t i = 0; i < device_info_count_; ++i) {
    devices_[i] = new DeviceInstance(i, *this, &device_infos_[i]);
  }

  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// Top-level API binding.
//===----------------------------------------------------------------------===//

void BindMonomorphicApi(PJRT_Api* api, Globals& globals) {
  api->struct_size = PJRT_Api_STRUCT_SIZE;
  api->priv = &globals;

  // Bind by object types.
  ClientInstance::BindApi(api);
  DeviceInstance::BindApi(api);
  ErrorInstance::BindApi(api);
}

}  // namespace iree::pjrt
