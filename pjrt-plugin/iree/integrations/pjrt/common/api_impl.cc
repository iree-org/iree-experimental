// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/integrations/pjrt/common/api_impl.h"

#include <iostream>
#include <optional>

#include "iree/hal/api.h"

namespace iree::pjrt {

// Chopped down utilities from various TPU support libraries. Basically all for
// populating Trimmed device shapes. Since that is supposed to go away at
// some point, just copy-pasta here.
namespace ApiConverter {
// Helper functions for copying data to possibly-inlined C arrays.

// 'Src' and 'Dst' are allowed to be different types to make this usable with
// memory-identical types, e.g. int64_t and int64_t. This should not be used
// with types that require a static_cast.
template <typename Src, typename Dst, typename DstList>
static void CreateVectorBase(const absl::Span<Src> src, DstList* dst) {
  dst->size = src.size();
  if (dst->size > TPU_C_API_MAX_INLINED) {
    dst->heap = new Dst[dst->size];
    std::copy(src.begin(), src.end(), dst->heap);
  } else {
    std::copy(src.begin(), src.end(), dst->inlined);
  }
}

void CreateVector(const absl::Span<const int64_t> src, Int64List* dst) {
  return CreateVectorBase<const int64_t, int64_t, Int64List>(src, dst);
}

void CreateVector(const absl::Span<const bool> src, BoolList* dst) {
  return CreateVectorBase<const bool, bool, BoolList>(src, dst);
}

static void CreateVector(const absl::Span<const bool> src, IntList* dst) {
  CreateVectorBase<const bool, int, IntList>(src, dst);
}

static void CreateVector(const absl::Span<const xla::DimLevelType> src,
                         IntList* dst) {
  CreateVectorBase<const xla::DimLevelType, int, IntList>(src, dst);
}

void ToC(const xla::Tile& tile, XLA_Tile* c_tile) {
  CreateVector(tile.dimensions(), &c_tile->dimensions);
}

static void CreateVector(const absl::Span<const xla::Tile> src, TileList* dst) {
  dst->size = src.size();
  XLA_Tile* c_tiles;
  if (dst->size > TPU_C_API_MAX_INLINED) {
    dst->heap = new XLA_Tile[dst->size];
    c_tiles = dst->heap;
  } else {
    c_tiles = dst->inlined;
  }
  for (int i = 0; i < dst->size; ++i) {
    ToC(src[i], &c_tiles[i]);
  }
}

void ToC(const xla::Layout& layout, XLA_Layout* c_layout) {
  CreateVector(layout.minor_to_major(), &c_layout->minor_to_major);
  CreateVector(layout.dim_level_types(), &c_layout->dim_level_types);
  CreateVector(layout.dim_unique(), &c_layout->dim_unique);
  CreateVector(layout.dim_ordered(), &c_layout->dim_ordered);
  c_layout->index_primitive_type = layout.index_primitive_type();
  c_layout->pointer_primitive_type = layout.pointer_primitive_type();
  c_layout->memory_space = layout.memory_space();
  CreateVector(layout.tiles(), &c_layout->tiles);
}

}  // namespace ApiConverter

namespace {

iree_status_t MapElementTypeToXlaElementType(
    iree_hal_element_type_t element_type, xla::PrimitiveType* xla_primitive) {
  // TODO: Cascade on bit-field sub-types to avoid large linear scan.
  switch (element_type) {
    // TODO: How do I interpret signless?
    case IREE_HAL_ELEMENT_TYPE_INT_8:
      *xla_primitive = xla::PrimitiveType::U8;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_INT_16:
      *xla_primitive = xla::PrimitiveType::U16;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_INT_32:
      *xla_primitive = xla::PrimitiveType::U32;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_INT_64:
      *xla_primitive = xla::PrimitiveType::U64;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_SINT_8:
      *xla_primitive = xla::PrimitiveType::S8;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_SINT_16:
      *xla_primitive = xla::PrimitiveType::S16;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_SINT_32:
      *xla_primitive = xla::PrimitiveType::S32;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_SINT_64:
      *xla_primitive = xla::PrimitiveType::S64;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_8:
      *xla_primitive = xla::PrimitiveType::U8;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_16:
      *xla_primitive = xla::PrimitiveType::U16;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_32:
      *xla_primitive = xla::PrimitiveType::U32;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_UINT_64:
      *xla_primitive = xla::PrimitiveType::U64;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_FLOAT_16:
      *xla_primitive = xla::PrimitiveType::F16;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_FLOAT_32:
      *xla_primitive = xla::PrimitiveType::U32;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_FLOAT_64:
      *xla_primitive = xla::PrimitiveType::F64;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
      *xla_primitive = xla::PrimitiveType::BF16;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_64:
      *xla_primitive = xla::PrimitiveType::C64;
      return iree_ok_status();
    case IREE_HAL_ELEMENT_TYPE_COMPLEX_FLOAT_128:
      *xla_primitive = xla::PrimitiveType::C128;
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

}  // namespace

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
        break;
      case IREE_STATUS_UNKNOWN:
        args->code = PJRT_Error_Code_UNKNOWN;
        break;
      case IREE_STATUS_INVALID_ARGUMENT:
        args->code = PJRT_Error_Code_INVALID_ARGUMENT;
        break;
      case IREE_STATUS_DEADLINE_EXCEEDED:
        args->code = PJRT_Error_Code_DEADLINE_EXCEEDED;
        break;
      case IREE_STATUS_NOT_FOUND:
        args->code = PJRT_Error_Code_NOT_FOUND;
        break;
      case IREE_STATUS_ALREADY_EXISTS:
        args->code = PJRT_Error_Code_ALREADY_EXISTS;
        break;
      case IREE_STATUS_PERMISSION_DENIED:
        args->code = PJRT_Error_Code_PERMISSION_DENIED;
        break;
      case IREE_STATUS_RESOURCE_EXHAUSTED:
        args->code = PJRT_Error_Code_RESOURCE_EXHAUSTED;
        break;
      case IREE_STATUS_FAILED_PRECONDITION:
        args->code = PJRT_Error_Code_FAILED_PRECONDITION;
        break;
      case IREE_STATUS_ABORTED:
        args->code = PJRT_Error_Code_ABORTED;
        break;
      case IREE_STATUS_OUT_OF_RANGE:
        args->code = PJRT_Error_Code_OUT_OF_RANGE;
        break;
      case IREE_STATUS_UNIMPLEMENTED:
        args->code = PJRT_Error_Code_UNIMPLEMENTED;
        break;
      case IREE_STATUS_INTERNAL:
        args->code = PJRT_Error_Code_INTERNAL;
        break;
      case IREE_STATUS_UNAVAILABLE:
        args->code = PJRT_Error_Code_UNAVAILABLE;
        break;
      case IREE_STATUS_DATA_LOSS:
        args->code = PJRT_Error_Code_DATA_LOSS;
        break;
      case IREE_STATUS_UNAUTHENTICATED:
        args->code = PJRT_Error_Code_UNAUTHENTICATED;
        break;
      case IREE_STATUS_DEFERRED:
        args->code = PJRT_Error_Code_UNKNOWN;  // No mapping
        break;
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

BufferInstance::~BufferInstance() = default;

iree_status_t BufferInstance::GetXlaShape(xla::Shape** out_shape) {
  if (cached_shape_) {
    *out_shape = &(*cached_shape_);
    return iree_ok_status();
  }

  iree_hal_element_type_t hal_element_type =
      iree_hal_buffer_view_element_type(buffer_view());
  xla::PrimitiveType xla_element_type;
  IREE_RETURN_IF_ERROR(
      MapElementTypeToXlaElementType(hal_element_type, &xla_element_type));

  size_t rank = iree_hal_buffer_view_shape_rank(buffer_view());
  const iree_hal_dim_t* dims = iree_hal_buffer_view_shape_dims(buffer_view());
  std::array<int64_t, 9> xla_dims;
  if (rank > xla_dims.size()) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "rank > 9 not supported");
  }
  for (size_t i = 0; i < rank; ++i) {
    xla_dims[i] = dims[i];
  }

  cached_shape_ = xla::ShapeUtil::MakeShape(
      xla_element_type,
      absl::MakeSpan(xla_dims.begin(), xla_dims.begin() + rank));
  *out_shape = &(*cached_shape_);
  return iree_ok_status();
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
      BufferInstance* buffer = BufferInstance::Unwrap(args->buffer);
      xla::Shape* shape;
      IREE_RETURN_IF_ERROR(buffer->GetXlaShape(&shape));

      args->element_type = shape->element_type();
      ApiConverter::CreateVector(shape->dimensions(), &args->dimensions);
      ApiConverter::CreateVector(shape->dynamic_dimensions(),
                                 &args->dynamic_dimensions);

      if (shape->has_layout()) {
        args->has_layout = true;
        ApiConverter::ToC(shape->layout(), &args->layout);
      } else {
        args->has_layout = false;
      }
      return iree_ok_status();
    };
    return MakeError(impl());
  };
  api->PJRT_Buffer_ToHostBuffer =
      +[](PJRT_Buffer_ToHostBuffer_Args* args) -> PJRT_Error* {
    BufferInstance* buffer = BufferInstance::Unwrap(args->src);
    if (!args->dst) {
      // Size query.
      return MakeError(buffer->GetHostSizeInBytes(&args->dst_size));
    } else {
      // Initiate transfer.
      return MakeError(
          buffer->CopyToHost(args->dst, args->dst_size,
                             reinterpret_cast<EventInstance**>(&args->event)));
    }
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
    args->is_deleted = BufferInstance::Unwrap(args->buffer)->is_deleted();
    return nullptr;
  };
  api->PJRT_Buffer_CopyToDevice =
      +[](PJRT_Buffer_CopyToDevice_Args* args) -> PJRT_Error* {
    return MakeError(iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                                      "PJRT_Buffer_CopyToDevice"));
  };
  api->PJRT_Buffer_IsOnCpu =
      +[](PJRT_Buffer_IsOnCpu_Args* args) -> PJRT_Error* {
    args->is_on_cpu = BufferInstance::Unwrap(args->buffer)->is_on_cpu();
    return nullptr;
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

iree_status_t BufferInstance::GetHostSizeInBytes(iree_host_size_t* host_size) {
  *host_size = iree_hal_buffer_view_byte_length(buffer_view());
  return iree_ok_status();
}

iree_status_t BufferInstance::CopyToHost(void* dst, iree_host_size_t dst_size,
                                         EventInstance** done_event) {
  // TODO: Do an async transfer on a transfer queue like a grown up.
  iree_hal_device_t* hal_device;
  IREE_RETURN_IF_ERROR(device_.GetHalDevice(&hal_device));
  IREE_RETURN_IF_ERROR(iree_hal_device_transfer_d2h(
      hal_device, iree_hal_buffer_view_buffer(buffer_view()), 0, dst, dst_size,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  *done_event = new EventInstance();
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// DeviceInstance
//===----------------------------------------------------------------------===//

DeviceInstance::~DeviceInstance() = default;

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
  iree_device_size_t element_type_byte_size =
      iree_hal_element_dense_byte_count(element_type);

  // Handle strided layouts.
  bool dense_row_major_layout = true;
  if (byte_strides && num_dims > 0) {
    int64_t stride = element_type_byte_size;
    for (int64_t i = num_dims - 1; i >= 0; --i) {
      if (byte_strides[i] != stride) {
        dense_row_major_layout = false;
        break;
      }
      stride *= dims[i];
    }
  }
  if (!dense_row_major_layout) {
    // TODO: Compile a transpose program and invoke that to load the
    // array.
    return iree_make_status(
        IREE_STATUS_UNIMPLEMENTED,
        "only dense, row-major layouts currently supported");
  }

  // Compute dense size.
  std::array<iree_hal_dim_t, 9> shape;
  if (num_dims > shape.size()) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "only supports up to %d dims but got %d",
                            (int)shape.size(), (int)num_dims);
  }

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
      iree_hal_device_allocator(device_.get()), num_dims, &shape[0],
      element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, params,
      iree_make_const_byte_span(data, byte_length), &buffer_view));

  // Since we synchronously copied, return an already signalled event.
  *out_done_with_host_buffer_event = new EventInstance();

  // Construct and return a BufferInstance.
  *out_buffer = new BufferInstance(*this, buffer_view);

  return iree_ok_status();
}

iree_status_t DeviceInstance::GetHalDevice(iree_hal_device_t** out_device) {
  IREE_RETURN_IF_ERROR(OpenDevice());
  *out_device = device_.get();
  return iree_ok_status();
}

//===----------------------------------------------------------------------===//
// ClientInstance
//===----------------------------------------------------------------------===//

ClientInstance::ClientInstance(std::unique_ptr<Platform> platform)
    : platform_(std::move(platform)) {
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
  // Explicitly releasing vs using a ref so as to better control shut-down
  // ordering (bad shutdown ordering of the driver is a frequent cause of
  // bugs).
  iree_hal_driver_release(driver_);
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
  // TODO: Remove calls to iree_status_fprint once JAX properly reports
  // initialization errors: https://github.com/google/jax/issues/13763
  auto status = CreateDriver(&driver_);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return MakeError(status);
  }

  status = InitializeVM();
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return MakeError(status);
  }

  status = PopulateDevices();
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    return MakeError(status);
  }

  // More initialization.
  return nullptr;
}

iree_status_t ClientInstance::InitializeVM() {
  IREE_RETURN_IF_ERROR(iree_vm_instance_create(host_allocator_, &vm_instance_));
  IREE_RETURN_IF_ERROR(iree_hal_module_register_all_types(vm_instance_.get()));
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
                                    ExecutableInstance** out_executable) {
  std::unique_ptr<ArtifactDumper::Transaction> artifact_tx;
  if (platform().artifact_dumper().enabled()) {
    artifact_tx = platform().artifact_dumper().CreateTransaction();
  }

  iree_status_t status;
  std::string_view format(program->format, program->format_size);
  std::string_view code(program->code, program->code_size);
  if (artifact_tx) {
    artifact_tx->WriteArtifact(/*label=*/"program", /*extension=*/"mlir",
                               /*index=*/-1, code);
  }

  if (format != "mlir") {
    // See: https://github.com/google/jax/issues/13722
    return MakeError(iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "because IREE only supports MLIR input but got something else"));
  }

  std::unique_ptr<CompilerJob> job = platform().compiler().StartJob();
  if (artifact_tx) {
    job->EnableCrashDumps(artifact_tx.get());
  }
  auto MakeCompilerError = [&]() {
    std::string message = job->GetErrorMessage();
    return MakeError(iree_make_status(IREE_STATUS_INVALID_ARGUMENT, ": %s",
                                      message.c_str()));
  };

  // Set flags.
  // TODO: This should be done as part of session setup from a named pool.
  // TODO: The HAL backends and other flags should come from the assigned
  // devices.
  if (!job->SetFlag("--iree-input-type=mhlo")) {
    return MakeCompilerError();
  }
  if (!SetDefaultCompilerFlags(job.get())) {
    return MakeCompilerError();
  }
  if (artifact_tx) {
    artifact_tx->WriteArtifact(
        /*label=*/"flags", /*extension=*/"txt", /*index=*/-1, job->GetFlags());
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
  if (artifact_tx) {
    artifact_tx->WriteArtifact(
        /*label=*/"program", /*extension=*/"vmfb", /*index=*/-1,
        std::string_view(static_cast<const char*>(output->GetData()),
                         output->GetDataSize()));
  }

  auto executable = std::make_unique<ExecutableInstance>(
      *this, std::move(output), addressable_devices_);
  status = executable->LoadAll();
  if (!iree_status_is_ok(status)) {
    return MakeError(status);
  }

  *out_executable = executable.release();

  if (artifact_tx) {
    artifact_tx->Cancel();
  }
  return nullptr;
}

iree_status_t ClientInstance::PopulateVMModules(
    std::vector<iree::vm::ref<iree_vm_module_t>>& modules,
    iree_hal_device_t* hal_device,
    iree::vm::ref<iree_vm_module_t>& main_module) {
  // HAL module.
  modules.push_back({});
  IREE_RETURN_IF_ERROR(iree_hal_module_create(
      vm_instance(), hal_device, IREE_HAL_MODULE_FLAG_NONE, host_allocator(),
      &modules.back()));

  // Main module.
  modules.push_back(main_module);
  return iree_ok_status();
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

iree_status_t EventInstance::OnReady(PJRT_Event_OnReadyCallback callback,
                                     void* user_arg) {
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
        ExecutableInstance::Unwrap(args->executable)->BatchExecute(args));
  };
  api->PJRT_Executable_NumOutputs =
      +[](PJRT_Executable_NumOutputs_Args* args) -> PJRT_Error* {
    auto* exec = ExecutableInstance::Unwrap(args->executable);
    iree_host_size_t arg_count;
    iree_host_size_t result_count;
    auto status = exec->GetArgResultCount(&arg_count, &result_count);
    args->num_outputs = result_count;
    return MakeError(status);
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

iree_status_t ExecutableInstance::LoadAll() {
  if (!loaded_executables_.empty()) return iree_ok_status();

  std::vector<LoadedExecutable> new_list;
  for (DeviceInstance* device_instance : addressable_devices_) {
    iree_hal_device_t* hal_device;
    IREE_RETURN_IF_ERROR(device_instance->GetHalDevice(&hal_device));
    new_list.push_back({});
    LoadedExecutable& loaded = new_list.back();
    loaded.device_instance = device_instance;

    IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
        client_.vm_instance(),
        iree_make_const_byte_span(binary_->GetData(), binary_->GetDataSize()),
        /*archive_allocator=*/iree_allocator_null(), client_.host_allocator(),
        &loaded.main_module));

    // Lookup main function.
    const char kNameMain[] = "main";
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_name(
        loaded.main_module.get(), IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_string_view_t{kNameMain, sizeof(kNameMain) - 1},
        &loaded.main_function));

    // Record number of args/results.
    iree_vm_function_signature_t sig =
        iree_vm_function_signature(&loaded.main_function);
    IREE_RETURN_IF_ERROR(iree_vm_function_call_count_arguments_and_results(
        &sig, &loaded.arg_count, &loaded.result_count));

    // Defer to the client to populate the stack of modules.
    std::vector<iree::vm::ref<iree_vm_module_t>> modules;
    IREE_RETURN_IF_ERROR(
        client_.PopulateVMModules(modules, hal_device, loaded.main_module));
    std::vector<iree_vm_module_t*> module_ptrs;
    module_ptrs.resize(modules.size());
    for (size_t i = 0; i < modules.size(); ++i) {
      module_ptrs[i] = modules[i].get();
    }

    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        client_.vm_instance(), IREE_VM_CONTEXT_FLAG_NONE, module_ptrs.size(),
        module_ptrs.data(), iree_allocator_system(), &loaded.vm_context));
  }

  new_list.swap(loaded_executables_);
  return iree_ok_status();
}

iree_status_t ExecutableInstance::GetDefaultLoadedExecutable(
    LoadedExecutable** out_loaded) {
  IREE_RETURN_IF_ERROR(LoadAll());
  if (loaded_executables_.empty()) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "no executables could be loaded");
  }
  *out_loaded = &loaded_executables_.front();
  return iree_ok_status();
}

iree_status_t ExecutableInstance::GetArgResultCount(
    iree_host_size_t* out_arg_count, iree_host_size_t* out_result_count) {
  LoadedExecutable* loaded;
  IREE_RETURN_IF_ERROR(GetDefaultLoadedExecutable(&loaded));
  *out_arg_count = loaded->arg_count;
  *out_result_count = loaded->result_count;
  return iree_ok_status();
}

iree_status_t ExecutableInstance::BatchExecute(
    PJRT_Executable_Execute_Args* args) {
  // Early exit for unsupported features and illegal input.
  if (args->execute_device) {
    return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                            "executing with a specific device not supported");
  }
  if (args->num_devices != addressable_devices_.size()) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "incorrect number of devices to execute on (%d vs %d)",
        (int)args->num_devices, (int)addressable_devices_.size());
  }

  // Make sure loaded.
  IREE_RETURN_IF_ERROR(LoadAll());

  // Initialize invocations.
  auto allocator = client_.host_allocator();
  auto& loaded_execs = loaded_executables_;
  struct Invocation {
    LoadedExecutable* dev_exe;
    iree::vm::ref<iree_vm_list_t> inputs;
    iree::vm::ref<iree_vm_list_t> outputs;
  };
  std::vector<Invocation> invs;
  invs.resize(args->num_devices);
  for (size_t dev_index = 0; dev_index < args->num_devices; ++dev_index) {
    auto& inv = invs[dev_index];
    inv.dev_exe = &loaded_execs[dev_index];
    IREE_RETURN_IF_ERROR(iree_vm_list_create(
        /*element_type=*/nullptr, args->num_args, allocator, &inv.inputs));
    IREE_RETURN_IF_ERROR(iree_vm_list_create(
        /*element_type=*/nullptr, inv.dev_exe->result_count, allocator,
        &inv.outputs));

    // Populate inputs.
    for (size_t i = 0; i < args->num_args; ++i) {
      auto* buffer = BufferInstance::Unwrap(args->argument_lists[dev_index][i]);
      iree_vm_ref_t bv_ref =
          iree_hal_buffer_view_retain_ref(buffer->buffer_view());
      IREE_RETURN_IF_ERROR(
          iree_vm_list_push_ref_move(inv.inputs.get(), &bv_ref));
    }
  }

  // Issue invocations.
  // TODO: Switch to using the async API. I've tried to structure this
  // so that we can move to that. Obviously important before we have more
  // than one device.
  iree_status_t status = iree_ok_status();
  for (size_t dev_index = 0; dev_index < args->num_devices; ++dev_index) {
    auto& inv = invs[dev_index];
    status = iree_vm_invoke(
        inv.dev_exe->vm_context.get(), inv.dev_exe->main_function,
        IREE_VM_INVOCATION_FLAG_NONE,
        /*policy=*/nullptr, inv.inputs.get(), inv.outputs.get(), allocator);
    if (!iree_status_is_ok(status)) break;
  }

  // Process results.
  // Early exit before committing things to the client if anything failed.
  if (!iree_status_is_ok(status)) return status;
  for (size_t dev_index = 0; dev_index < args->num_devices; ++dev_index) {
    auto& inv = invs[dev_index];
    for (size_t i = 0; i < inv.dev_exe->result_count; ++i) {
      iree_hal_buffer_view_t* ret_buffer_view =
          (iree_hal_buffer_view_t*)iree_vm_list_get_ref_deref(
              inv.outputs.get(), i, iree_hal_buffer_view_get_descriptor());
      // This should not be possible so just hard-assert.
      IREE_ASSERT_ARGUMENT(ret_buffer_view);
      iree_hal_buffer_view_retain(ret_buffer_view);
      args->output_lists[dev_index][i] =
          *(new BufferInstance(*inv.dev_exe->device_instance, ret_buffer_view));
    }

    if (args->device_complete_events) {
      args->device_complete_events[dev_index] = *(new EventInstance());
    }
  }

  return status;
}

//===----------------------------------------------------------------------===//
// Top-level API binding.
//===----------------------------------------------------------------------===//

void BindMonomorphicApi(PJRT_Api* api) {
  api->struct_size = PJRT_Api_STRUCT_SIZE;
  api->priv = nullptr;

  // Bind by object types.
  BufferInstance::BindApi(api);
  ClientInstance::BindApi(api);
  DeviceInstance::BindApi(api);
  ErrorInstance::BindApi(api);
  EventInstance::BindApi(api);
  ExecutableInstance::BindApi(api);
}

}  // namespace iree::pjrt
