// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/integrations/pjrt/common/api_impl.h"

#include <iostream>

#include "iree/hal/buffer_view.h"

namespace iree::pjrt {

namespace {

iree_status_t MapElementTypeToBufferType(iree_hal_element_type_t element_type,
                                         PJRT_Buffer_Type* buffer_type) {
  switch (element_type) {
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
      *buffer_type = PJRT_Buffer_Type_S8;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
      *buffer_type = PJRT_Buffer_Type_S16;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
      *buffer_type = PJRT_Buffer_Type_S32;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
      *buffer_type = PJRT_Buffer_Type_S64;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      *buffer_type = PJRT_Buffer_Type_U8;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      *buffer_type = PJRT_Buffer_Type_U16;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      *buffer_type = PJRT_Buffer_Type_U32;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      *buffer_type = PJRT_Buffer_Type_U64;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      *buffer_type = PJRT_Buffer_Type_F16;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      *buffer_type = PJRT_Buffer_Type_F32;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      *buffer_type = PJRT_Buffer_Type_F64;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
      *buffer_type = PJRT_Buffer_Type_BF16;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64:
      *buffer_type = PJRT_Buffer_Type_C64;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128:
      *buffer_type = PJRT_Buffer_Type_C128;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "conversion from unknown element type 0x%x",
                              (int)element_type);
  }
}

iree_status_t MapBufferTypeToElementType(
    PJRT_Buffer_Type buffer_type, iree_hal_element_type_t* element_type) {
  switch (buffer_type) {
    case PJRT_Buffer_Type_INVALID:
      return iree_make_status(IREE_STATUS_INVALID_ARGUMENT);
    case PJRT_Buffer_Type_PRED:
      // I assume this is equiv to IREE_HAL_ELEMENT_TYPE_BOOL_8 but
      // need to check.
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "TODO: Support PRED buffer type");
    case PJRT_Buffer_Type_S8:
      *element_type = IREE_HAL_ELEMENT_TYPE_SINT_8;
      return iree_ok_status();
    case PJRT_Buffer_Type_S16:
      *element_type = IREE_HAL_ELEMENT_TYPE_SINT_16;
      return iree_ok_status();
    case PJRT_Buffer_Type_S32:
      *element_type = IREE_HAL_ELEMENT_TYPE_SINT_32;
      return iree_ok_status();
    case PJRT_Buffer_Type_S64:
      *element_type = IREE_HAL_ELEMENT_TYPE_SINT_64;
      return iree_ok_status();
    case PJRT_Buffer_Type_U8:
      *element_type = IREE_HAL_ELEMENT_TYPE_UINT_8;
      return iree_ok_status();
    case PJRT_Buffer_Type_U16:
      *element_type = IREE_HAL_ELEMENT_TYPE_UINT_16;
      return iree_ok_status();
    case PJRT_Buffer_Type_U32:
      *element_type = IREE_HAL_ELEMENT_TYPE_UINT_32;
      return iree_ok_status();
    case PJRT_Buffer_Type_U64:
      *element_type = IREE_HAL_ELEMENT_TYPE_UINT_64;
      return iree_ok_status();
    case PJRT_Buffer_Type_F16:
      *element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_16;
      return iree_ok_status();
    case PJRT_Buffer_Type_F32:
      *element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_32;
      return iree_ok_status();
    case PJRT_Buffer_Type_F64:
      *element_type = IREE_HAL_ELEMENT_TYPE_FLOAT_64;
      return iree_ok_status();
    case PJRT_Buffer_Type_BF16:
      *element_type = IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
      return iree_ok_status();
    case PJRT_Buffer_Type_C64:
      *element_type = IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64;
      return iree_ok_status();
    case PJRT_Buffer_Type_C128:
      *element_type = IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "conversion from unknown buffer type %d",
                              (int)buffer_type);
  }
}

iree_status_t MapBufferTypeToXlaElementType(PJRT_Buffer_Type buffer_type,
                                            int* xla_element_type) {
  // See xla_data.proto:PrimitiveType.
  // This should go away once the bug is fixed and exposed properly
  // through the C API.
  switch (buffer_type) {
    case PJRT_Buffer_Type_INVALID:
      *xla_element_type = 0;
      return iree_ok_status();
    case PJRT_Buffer_Type_PRED:
      *xla_element_type = 1;
      return iree_ok_status();
    case PJRT_Buffer_Type_S8:
      *xla_element_type = 2;
      return iree_ok_status();
    case PJRT_Buffer_Type_S16:
      *xla_element_type = 3;
      return iree_ok_status();
    case PJRT_Buffer_Type_S32:
      *xla_element_type = 4;
      return iree_ok_status();
    case PJRT_Buffer_Type_S64:
      *xla_element_type = 5;
      return iree_ok_status();
    case PJRT_Buffer_Type_U8:
      *xla_element_type = 6;
      return iree_ok_status();
    case PJRT_Buffer_Type_U16:
      *xla_element_type = 7;
      return iree_ok_status();
    case PJRT_Buffer_Type_U32:
      *xla_element_type = 8;
      return iree_ok_status();
    case PJRT_Buffer_Type_U64:
      *xla_element_type = 9;
      return iree_ok_status();
    case PJRT_Buffer_Type_F16:
      *xla_element_type = 10;
      return iree_ok_status();
    case PJRT_Buffer_Type_F32:
      *xla_element_type = 11;
      return iree_ok_status();
    case PJRT_Buffer_Type_F64:
      *xla_element_type = 12;
      return iree_ok_status();
    case PJRT_Buffer_Type_BF16:
      *xla_element_type = 16;
      return iree_ok_status();
    case PJRT_Buffer_Type_C64:
      *xla_element_type = 15;
      return iree_ok_status();
    case PJRT_Buffer_Type_C128:
      *xla_element_type = 18;
      return iree_ok_status();
    default:
      return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                              "conversion from unknown buffer type %d",
                              (int)buffer_type);
  }
}

}  // namespace

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
// BufferInstance
//===----------------------------------------------------------------------===//

BufferInstance::~BufferInstance() {
  iree_hal_buffer_view_release(buffer_view_);
}

void BufferInstance::BindApi(PJRT_Api* api) {
  api->PJRT_Buffer_Destroy =
      +[](PJRT_Buffer_Destroy_Args* args) -> PJRT_Error* {
    delete BufferInstance::Unwrap(args->buffer);
    return nullptr;
  };
  api->PJRT_Buffer_OnDeviceTrimmedShape =
      +[](PJRT_Buffer_OnDeviceTrimmedShape_Args* args) -> PJRT_Error* {
    auto impl = [&]() -> iree_status_t {
      // TODO: This function is terrible and not exposed properly to C.
      // It is slated to be deleted...
      // See Google bug b/238999986
      auto* bv = BufferInstance::Unwrap(args->buffer)->buffer_view();
      iree_hal_element_type_t hal_element_type =
          iree_hal_buffer_view_element_type(bv);
      PJRT_Buffer_Type pjrt_buffer_type;
      IREE_RETURN_IF_ERROR(
          MapElementTypeToBufferType(hal_element_type, &pjrt_buffer_type));
      IREE_RETURN_IF_ERROR(
          MapBufferTypeToXlaElementType(pjrt_buffer_type, &args->element_type));

      args->has_layout = false;

      auto rank = iree_hal_buffer_view_shape_rank(bv);
      const iree_hal_dim_t* dims = iree_hal_buffer_view_shape_dims(bv);

      int64_t* dim_list = args->dimensions.inlined;
      bool* dyn_list = args->dynamic_dimensions.inlined;
      if (rank > TPU_C_API_MAX_INLINED) {
        dim_list = args->dimensions.heap =
            (int64_t*)malloc(sizeof(int64_t) * rank);
        dyn_list = args->dynamic_dimensions.heap =
            (bool*)malloc(sizeof(int64_t) * rank);
      }
      for (size_t i = 0; i < rank; ++i) {
        dim_list[i] = dims[i];
        dyn_list[i] = false;
      }
      args->dimensions.size = rank;
      args->dynamic_dimensions.size = rank;
      return iree_ok_status();
    };
    return MakeError(impl());
  };
  api->PJRT_Buffer_ToHostBuffer =
      +[](PJRT_Buffer_ToHostBuffer_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Buffer_ToHostBuffer"));
  };
  api->PJRT_Buffer_OnDeviceSizeInBytes =
      +[](PJRT_Buffer_OnDeviceSizeInBytes_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Buffer_OnDeviceSizeInBytes"));
  };
  api->PJRT_Buffer_Delete = +[](PJRT_Buffer_Delete_Args* args) -> PJRT_Error* {
    return MakeError(
        iree_make_status(IREE_STATUS_UNIMPLEMENTED, "PJRT_Buffer_Delete"));
  };
  api->PJRT_Buffer_IsDeleted =
      +[](PJRT_Buffer_IsDeleted_Args* args) -> PJRT_Error* {
    return MakeError(
        iree_make_status(IREE_STATUS_UNIMPLEMENTED, "PJRT_Buffer_IsDeleted"));
  };
  api->PJRT_Buffer_CopyToDevice =
      +[](PJRT_Buffer_CopyToDevice_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Buffer_CopyToDevice"));
  };
  api->PJRT_Buffer_IsOnCpu =
      +[](PJRT_Buffer_IsOnCpu_Args* args) -> PJRT_Error* {
    return MakeError(
        iree_make_status(IREE_STATUS_UNIMPLEMENTED, "PJRT_Buffer_IsOnCpu"));
  };
  api->PJRT_Buffer_Device = +[](PJRT_Buffer_Device_Args* args) -> PJRT_Error* {
    args->device = BufferInstance::Unwrap(args->buffer)->device();
    return nullptr;
  };
  api->PJRT_Buffer_ReadyEvent =
      +[](PJRT_Buffer_ReadyEvent_Args* args) -> PJRT_Error* {
    return MakeError(
        iree_make_status(IREE_STATUS_UNIMPLEMENTED, "PJRT_Buffer_ReadyEvent"));
  };
  api->PJRT_Buffer_UnsafePointer =
      +[](PJRT_Buffer_UnsafePointer_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Buffer_UnsafePointer"));
  };
}

//===----------------------------------------------------------------------===//
// DeviceInstance
//===----------------------------------------------------------------------===//

DeviceInstance::~DeviceInstance() {
  if (device_) {
    iree_hal_device_release(device_);
  }
}

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
  api->PJRT_Device_Kind = +[](PJRT_Device_Kind_Args* args) -> PJRT_Error* {
    return MakeError(
        iree_make_status(IREE_STATUS_UNIMPLEMENTED, "PJRT_Device_Kind"));
  };
  api->PJRT_Device_LocalHardwareId =
      +[](PJRT_Device_LocalHardwareId_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Device_LocalHardwareId"));
  };
  api->PJRT_Device_DebugString =
      +[](PJRT_Device_DebugString_Args* args) -> PJRT_Error* {
    return MakeError(
        iree_make_status(IREE_STATUS_UNIMPLEMENTED, "PJRT_Device_DebugString"));
  };
  api->PJRT_Device_ToString =
      +[](PJRT_Device_ToString_Args* args) -> PJRT_Error* {
    return MakeError(
        iree_make_status(IREE_STATUS_UNIMPLEMENTED, "PJRT_Device_ToString"));
  };
}

iree_status_t DeviceInstance::OpenDevice() {
  if (device_) return iree_ok_status();
  return iree_hal_driver_create_device_by_id(
      driver_, /*device_id=*/info_->device_id,
      /*param_count=*/0, /*params=*/nullptr, client_.host_allocator(),
      &device_);
}

iree_status_t DeviceInstance::HostBufferToDevice(
    const void* data, PJRT_Buffer_Type type, const int64_t* dims,
    size_t num_dims, const int64_t* byte_strides, size_t num_byte_strides,
    PJRT_HostBufferSemantics host_buffer_semantics,
    EventInstance** out_done_with_host_buffer_event,
    BufferInstance** out_buffer) {
  IREE_RETURN_IF_ERROR(OpenDevice());

  // Early-exit on unimplemented features.
  if (byte_strides && num_dims > 0) {
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "because host buffers with strides are note yet implemented");
  }

  // Map element type.
  iree_hal_element_type_t element_type;
  IREE_RETURN_IF_ERROR(MapBufferTypeToElementType(type, &element_type));
  // TODO: Do something sensible with sub-byte aligned types.
  if (IREE_UNLIKELY(iree_hal_element_bit_count(element_type) == 0) ||
      IREE_UNLIKELY(!iree_hal_element_is_byte_aligned(element_type))) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "opaque and sub-byte aligned element types cannot be indexed");
  }

  // Compute dense size.
  std::array<iree_hal_dim_t, 9> shape;
  if (num_dims > shape.size()) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "only supports up to %d dims but got %d",
                            (int)shape.size(), (int)num_dims);
  }

  iree_device_size_t element_type_byte_size =
      iree_hal_element_dense_byte_count(element_type);
  iree_device_size_t byte_length = element_type_byte_size;
  for (size_t i = 0; i < num_dims; ++i) {
    shape[i] = dims[i];
    byte_length *= dims[i];
  }

  // TODO: Don't do synchronous h2d transfer. Instead issue a command against
  // the transfer queue like a grown-up. Also pay attention to zero copy flags
  // and such. Plenty to make efficient here.
  iree_hal_buffer_params_t params;
  memset(&params, 0, sizeof(params));
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;

  iree_hal_buffer_view_t* buffer_view = nullptr;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_allocate_buffer(
      iree_hal_device_allocator(device_), num_dims, &shape[0], element_type,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
      iree_make_const_byte_span(data, byte_length), &buffer_view));

  // Since we synchronously copied, return an already signalled event.
  *out_done_with_host_buffer_event = new EventInstance();

  // Construct and return a BufferInstance.
  *out_buffer = new BufferInstance(*this, buffer_view);

  return iree_ok_status();
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
    args->devices = const_cast<PJRT_Device**>(
        reinterpret_cast<PJRT_Device* const*>(devices.data()));
    args->num_devices = devices.size();
    return nullptr;
  };
  api->PJRT_Client_AddressableDevices =
      +[](PJRT_Client_AddressableDevices_Args* args) -> PJRT_Error* {
    auto& devices = ClientInstance::Unwrap(args->client)->addressable_devices();
    args->addressable_devices = const_cast<PJRT_Device**>(
        reinterpret_cast<PJRT_Device* const*>(devices.data()));
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
  api->PJRT_Client_Compile =
      +[](PJRT_Client_Compile_Args* args) -> PJRT_Error* {
    // TODO: It is not great that we only get a client here vs a list of
    // devices to consider (or something). The issue is that systems often
    // have unrelated devices that will not actually be scheduled and those
    // will very naturally have different tuning flags. We therefore have to
    // guess... which is an accident waiting to happen.
    // Looks like what I need is buried in the compile options... need to
    // work on that.
    auto* client = ClientInstance::Unwrap(args->client);
    ExecutableInstance* executable;
    auto* error = client->Compile(args->program, &executable);
    if (error) return error;
    args->executable = *executable;
    return nullptr;
  };
  api->PJRT_Client_DefaultDeviceAssignment =
      +[](PJRT_Client_DefaultDeviceAssignment_Args* args) -> PJRT_Error* {
    // TODO: Something sensible.
    for (size_t i = 0; i < args->default_assignment_size; ++i) {
      args->default_assignment[i] = 0;
    }
    return nullptr;
  };
  api->PJRT_Client_BufferFromHostBuffer =
      +[](PJRT_Client_BufferFromHostBuffer_Args* args) -> PJRT_Error* {
    auto status =
        DeviceInstance::Unwrap(args->device)
            ->HostBufferToDevice(
                args->data, args->type, args->dims, args->num_dims,
                args->byte_strides, args->num_byte_strides,
                args->host_buffer_semantics,
                reinterpret_cast<EventInstance**>(&args->done_with_host_buffer),
                reinterpret_cast<BufferInstance**>(&args->buffer));
    return MakeError(status);
  };
}

PJRT_Error* ClientInstance::Initialize() {
  auto status = CreateDriver(&driver_);
  if (!iree_status_is_ok(status)) return MakeError(status);

  status = PopulateDevices();
  if (!iree_status_is_ok(status)) return MakeError(status);

  status = InitializeCompiler();
  if (!iree_status_is_ok(status)) return MakeError(status);

  // More initialization.
  return nullptr;
}

iree_status_t ClientInstance::InitializeCompiler() {
  // TODO: This needs an overhaul obviously. Should be backed by a Platform
  // factory that can be more customized for different deployment scenarios
  // (i.e. static linking, etc).
  compiler_ = InprocessStubCompiler::Initialize(
      "/home/stella/src/iree-build/lib/libIREECompiler.so");
  if (!compiler_) {
    return iree_make_status(
        IREE_STATUS_UNAVAILABLE,
        "because the compiler shared library could not be loaded");
  }

  return iree_ok_status();
}

iree_status_t ClientInstance::PopulateDevices() {
  IREE_RETURN_IF_ERROR(iree_hal_driver_query_available_devices(
      driver_, host_allocator_, &device_info_count_, &device_infos_));
  devices_.resize(device_info_count_);
  for (iree_host_size_t i = 0; i < device_info_count_; ++i) {
    // Note that we assume one driver per client here.
    // But device is modeled with a driver in case if it ever becomes
    // more heterogenous.
    devices_[i] = new DeviceInstance(i, *this, driver_, &device_infos_[i]);
  }

  // For now, just make all devices addressable.
  addressable_devices_.reserve(devices_.size());
  for (auto* device : devices_) {
    addressable_devices_.push_back(device);
  }
  return iree_ok_status();
}

PJRT_Error* ClientInstance::Compile(PJRT_Program* program,
                                    ExecutableInstance** executable) {
  std::string_view format(program->format, program->format_size);
  std::string_view code(program->code, program->code_size);
  if (format != "mlir") {
    // See: https://github.com/google/jax/issues/13722
    return MakeError(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "because IREE only supports MLIR input but got something else"));
  }

  std::unique_ptr<CompilerJob> job = compiler_->StartJob();
  auto MakeCompilerError = [&]() {
    std::string message = job->GetErrorMessage();
    return MakeError(iree_make_status(IREE_STATUS_INVALID_ARGUMENT, ": %s",
                                      message.c_str()));
  };

  // Set flags.
  // TODO: This should be done as part of session setup from a named pool.
  // TODO: The HAL backends and other flags should come from the assigned
  // devices.
  if (!job->SetFlag("--iree-input-type=mhlo") ||
      !job->SetFlag("--iree-hal-target-backends=llvm-cpu")) {
    return MakeCompilerError();
  }

  // Parse the source.
  if (!job->ParseSourceBuffer(code.data(), code.size())) {
    return MakeCompilerError();
  }

  // Perform main compilation.
  std::unique_ptr<CompilerOutput> output = job->CompileStandardPipeline();
  if (!output) {
    return MakeCompilerError();
  }

  *executable =
      new ExecutableInstance(*this, std::move(output), addressable_devices_);
  return nullptr;
}

//===----------------------------------------------------------------------===//
// EventInstance
//===----------------------------------------------------------------------===//

void EventInstance::BindApi(PJRT_Api* api) {
  api->PJRT_Event_Destroy = +[](PJRT_Event_Destroy_Args* args) -> PJRT_Error* {
    delete EventInstance::Unwrap(args->event);
    return nullptr;
  };
  api->PJRT_Event_IsReady = +[](PJRT_Event_IsReady_Args* args) -> PJRT_Error* {
    args->is_ready = EventInstance::Unwrap(args->event)->is_ready();
    return nullptr;
  };
  api->PJRT_Event_Error = +[](PJRT_Event_Error_Args* args) -> PJRT_Error* {
    return (PJRT_Error*)EventInstance::Unwrap(args->event)->error();
  };
  api->PJRT_Event_Await = +[](PJRT_Event_Await_Args* args) -> PJRT_Error* {
    return MakeError(
        iree_make_status(IREE_STATUS_UNIMPLEMENTED, "PJRT_Event_Await"));
  };
  api->PJRT_Event_OnReady = +[](PJRT_Event_OnReady_Args* args) -> PJRT_Error* {
    return MakeError(EventInstance::Unwrap(args->event)
                         ->OnReady(args->callback, args->user_arg));
  };
}

iree_status_t EventInstance::OnReady(PJRT_Event_OnReadyCallback callback, void* user_arg) {
  // TODO: Detect if not ready.
  callback((PJRT_Error*)error_, user_arg);
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// ExecutableInstance
//===----------------------------------------------------------------------===//

void ExecutableInstance::BindApi(PJRT_Api* api) {
  api->PJRT_Executable_Destroy =
      +[](PJRT_Executable_Destroy_Args* args) -> PJRT_Error* {
    delete ExecutableInstance::Unwrap(args->executable);
    return nullptr;
  };
  api->PJRT_Executable_Name =
      +[](PJRT_Executable_Name_Args* args) -> PJRT_Error* {
    const char* dummy_name = "iree_vmfb";
    args->executable_name = dummy_name;
    args->executable_name_size = strlen(dummy_name);
    return nullptr;
  };
  api->PJRT_Executable_AddressableDevices =
      +[](PJRT_Executable_AddressableDevices_Args* args) -> PJRT_Error* {
    auto& devices =
        ExecutableInstance::Unwrap(args->executable)->addressable_devices();
    args->addressable_devices = const_cast<PJRT_Device**>(
        reinterpret_cast<PJRT_Device* const*>(devices.data()));
    args->num_addressable_devices = devices.size();
    return nullptr;
  };
  api->PJRT_Executable_OptimizedProgram =
      +[](PJRT_Executable_OptimizedProgram_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Executable_OptimizedProgram"));
  };
  api->PJRT_Executable_Delete =
      +[](PJRT_Executable_Delete_Args* args) -> PJRT_Error* {
    return MakeError(
        iree_make_status(IREE_STATUS_UNIMPLEMENTED, "PJRT_Executable_Delete"));
  };
  api->PJRT_Executable_IsDeleted =
      +[](PJRT_Executable_IsDeleted_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Executable_IsDeleted"));
  };
  api->PJRT_Executable_Execute =
      +[](PJRT_Executable_Execute_Args* args) -> PJRT_Error* {
    return MakeError(
        iree_make_status(IREE_STATUS_UNIMPLEMENTED, "PJRT_Executable_Execute"));
  };
  api->PJRT_Executable_NumOutputs =
      +[](PJRT_Executable_NumOutputs_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Executable_NumOutputs"));
  };
  api->PJRT_Executable_SizeOfGeneratedCodeInBytes =
      +[](PJRT_Executable_SizeOfGeneratedCodeInBytes_Args* args)
      -> PJRT_Error* {
    return MakeError(
        iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                         "PJRT_Executable_SizeOfGeneratedCodeInBytes_Args"));
  };
  api->PJRT_Executable_GetCostAnalysis =
      +[](PJRT_Executable_GetCostAnalysis_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Executable_GetCostAnalysis"));
  };
  api->PJRT_Executable_Serialize =
      +[](PJRT_Executable_Serialize_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Executable_Serialize"));
  };
  api->PJRT_Executable_Deserialize =
      +[](PJRT_Executable_Deserialize_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Executable_Deserialize"));
  };
}

//===----------------------------------------------------------------------===//
// Top-level API binding.
//===----------------------------------------------------------------------===//

void BindMonomorphicApi(PJRT_Api* api, Globals& globals) {
  api->struct_size = PJRT_Api_STRUCT_SIZE;
  api->priv = &globals;

  // Bind by object types.
  BufferInstance::BindApi(api);
  ClientInstance::BindApi(api);
  DeviceInstance::BindApi(api);
  ErrorInstance::BindApi(api);
  EventInstance::BindApi(api);
  ExecutableInstance::BindApi(api);
}

}  // namespace iree::pjrt
