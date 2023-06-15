// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/async/module.h"

#include <thread>
#include <vector>

#include "iree/base/api.h"
#include "iree/base/internal/wait_handle.h"
#include "iree/base/loop_sync.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/testing/gtest.h"
#include "iree/testing/status_matchers.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"
#include "openxla/runtime/async/test/module_test_module_c.h"
#include "openxla/runtime/async_test/module.h"

iree_status_t async_callback(void* user_data, iree_loop_t loop,
                             iree_status_t status, iree_vm_list_t* outputs) {
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "Async invocation finished\n");
    fflush(stdout);
  } else {
    fprintf(stdout, "Async invocation failed\n");
    fflush(stdout);
  }
  iree_status_t* invoke_status = (iree_status_t*)user_data;
  *invoke_status = status;
  iree_vm_list_release(outputs);  // must be released!
  return iree_ok_status();
}

void AllocateLoop(iree_status_t* out_status, iree_allocator_t allocator,
                  iree_loop_t* out_loop) {
  iree_loop_sync_options_t options = {0};
  options.max_queue_depth = 128;
  options.max_wait_count = 32;

  iree_loop_sync_t* loop_sync = NULL;
  IREE_CHECK_OK(iree_loop_sync_allocate(options, allocator, &loop_sync));

  iree_loop_sync_scope_t* scope = NULL;
  IREE_CHECK_OK(
      iree_allocator_malloc(allocator, sizeof(*scope), (void**)&scope));
  iree_loop_sync_scope_initialize(
      loop_sync,
      +[](void* user_data, iree_status_t status) {
        iree_status_t* status_ptr = (iree_status_t*)user_data;
        if (iree_status_is_ok(*status_ptr)) {
          *status_ptr = status;
        } else {
          iree_status_ignore(status);
        }
      },
      out_status, scope);
  *out_loop = iree_loop_sync_scope(scope);
}

void FreeLoop(iree_allocator_t allocator, iree_loop_t loop) {
  iree_loop_sync_scope_t* scope = (iree_loop_sync_scope_t*)loop.self;
  iree_loop_sync_t* loop_sync = scope->loop_sync;

  iree_loop_sync_scope_deinitialize(scope);
  iree_allocator_free(allocator, scope);

  iree_loop_sync_free(loop_sync);
}

template <size_t N>
static std::vector<iree_vm_value_t> MakeValuesList(const int32_t (&values)[N]) {
  std::vector<iree_vm_value_t> result;
  result.resize(N);
  for (size_t i = 0; i < N; ++i) result[i] = iree_vm_value_make_i32(values[i]);
  return result;
}

static std::vector<iree_vm_ref_t> MakeHalBufferViewList(
    std::vector<std::string> values) {
  std::vector<iree_vm_ref_t> result;
  iree_hal_allocator_t* device_allocator;
  iree_hal_allocator_create_heap(iree_make_cstring_view("host_local"),
                                 iree_allocator_system(),
                                 iree_allocator_system(), &device_allocator);

  for (auto& buffer_value : values) {
    iree_hal_buffer_view_t* input_view;
    iree_hal_buffer_view_parse(
        iree_string_view_t{buffer_value.data(), buffer_value.size()},
        device_allocator, &input_view);
    result.push_back(iree_hal_buffer_view_retain_ref(input_view));
  }

  return result;
}

static bool operator==(const iree_vm_value_t& lhs,
                       const iree_vm_value_t& rhs) noexcept {
  if (lhs.type != rhs.type) return false;
  switch (lhs.type) {
    default:
    case IREE_VM_VALUE_TYPE_NONE:
      return true;  // none == none
    case IREE_VM_VALUE_TYPE_I8:
      return lhs.i8 == rhs.i8;
    case IREE_VM_VALUE_TYPE_I16:
      return lhs.i16 == rhs.i16;
    case IREE_VM_VALUE_TYPE_I32:
      return lhs.i32 == rhs.i32;
    case IREE_VM_VALUE_TYPE_I64:
      return lhs.i64 == rhs.i64;
    case IREE_VM_VALUE_TYPE_F32:
      return lhs.f32 == rhs.f32;
    case IREE_VM_VALUE_TYPE_F64:
      return lhs.f64 == rhs.f64;
  }
}

iree::StatusOr<std::string> BufferViewToString(
    iree_hal_buffer_view_t* buffer_view) {
  std::string result_str(4096, '\0');
  iree_status_t status;
  do {
    iree_host_size_t actual_length = 0;
    status = iree_hal_buffer_view_format(
        buffer_view, /*max_element_count=*/1024, result_str.size() + 1,
        &result_str[0], &actual_length);
    result_str.resize(actual_length);
  } while (iree_status_is_out_of_range(status));
  IREE_RETURN_IF_ERROR(std::move(status));
  return std::move(result_str);
}

static bool operator==(const iree_vm_ref_t& lhs,
                       const iree_vm_ref_t& rhs) noexcept {
  if (lhs.type == iree_hal_buffer_view_type() && lhs.type == rhs.type) {
    iree_hal_buffer_view_t* lbuffer = (iree_hal_buffer_view_t*)lhs.ptr;
    iree_hal_buffer_view_t* rbuffer = (iree_hal_buffer_view_t*)rhs.ptr;
    return *BufferViewToString(lbuffer) == *BufferViewToString(rbuffer);
  }
  return false;
}

namespace {

using iree::StatusCode;
using iree::StatusOr;
using iree::testing::status::IsOkAndHolds;
using iree::testing::status::StatusIs;
using iree::vm::ref;
using testing::Eq;

class AsyncRuntimeModuleTest : public ::testing::Test {
 protected:
  virtual void SetUp() {
    iree_allocator_t allocator = iree_allocator_system();

    IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                          allocator, &instance_));

    IREE_ASSERT_OK(iree_hal_module_register_all_types(instance_));

    IREE_CHECK_OK(iree_async_runtime_module_create(instance_, allocator,
                                                   &async_runtime_module_));
    IREE_CHECK_OK(openxla_async_test_module_create(instance_, allocator,
                                                   &async_test_module_));

    const auto* module_file_toc = openxla_async_module_test_module_create();
    IREE_CHECK_OK(iree_vm_bytecode_module_create(
        instance_,
        iree_const_byte_span_t{
            reinterpret_cast<const uint8_t*>(module_file_toc->data),
            module_file_toc->size},
        iree_allocator_null(), allocator, &bytecode_module_));

    iree_vm_module_t* modules[] = {async_runtime_module_, async_test_module_,
                                   bytecode_module_};

    IREE_CHECK_OK(iree_vm_context_create_with_modules(
        instance_, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), modules,
        allocator, &context_));
  }

  virtual void TearDown() {
    iree_vm_module_release(bytecode_module_);
    iree_vm_module_release(async_runtime_module_);
    iree_vm_module_release(async_test_module_);
    iree_vm_context_release(context_);
    iree_vm_instance_release(instance_);
  }

  StatusOr<std::vector<iree_vm_value_t>> RunFunction(
      const char* function_name, std::vector<iree_vm_value_t> inputs) {
    ref<iree_vm_list_t> input_list;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), inputs.size(),
                            iree_allocator_system(), &input_list));

    IREE_RETURN_IF_ERROR(iree_vm_list_resize(input_list.get(), inputs.size()));
    for (iree_host_size_t i = 0; i < inputs.size(); ++i) {
      IREE_RETURN_IF_ERROR(
          iree_vm_list_set_value(input_list.get(), i, &inputs[i]));
    }

    ref<iree_vm_list_t> output_list;
    IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             8, iree_allocator_system(),
                                             &output_list));

    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_name(
        bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_make_cstring_view(function_name), &function));

    iree_vm_async_invoke_state_t state = {};

    iree_loop_t loop;
    iree_status_t loop_status;
    AllocateLoop(&loop_status, iree_allocator_system(), &loop);

    iree_status_t invoke_status;
    iree_vm_async_invoke(
        loop, &state, context_, function, IREE_VM_INVOCATION_FLAG_NONE,
        /*policy=*/NULL, input_list.get(), output_list.get(),
        iree_allocator_system(), async_callback, &invoke_status);

    IREE_RETURN_IF_ERROR(iree_loop_drain(loop, iree_infinite_timeout()));
    IREE_RETURN_IF_ERROR(invoke_status);

    std::vector<iree_vm_value_t> outputs;
    outputs.resize(iree_vm_list_size(output_list.get()));
    for (iree_host_size_t i = 0; i < outputs.size(); ++i) {
      IREE_RETURN_IF_ERROR(
          iree_vm_list_get_value(output_list.get(), i, &outputs[i]));
    }

    FreeLoop(iree_allocator_system(), loop);
    return outputs;
  }

  StatusOr<std::vector<iree_vm_ref_t>> RunFunction(
      const char* function_name, std::vector<iree_vm_ref_t> inputs) {
    ref<iree_vm_list_t> input_list;
    IREE_RETURN_IF_ERROR(
        iree_vm_list_create(iree_vm_make_undefined_type_def(), inputs.size(),
                            iree_allocator_system(), &input_list));
    IREE_RETURN_IF_ERROR(iree_vm_list_resize(input_list.get(), inputs.size()));
    for (iree_host_size_t i = 0; i < inputs.size(); ++i) {
      IREE_RETURN_IF_ERROR(
          iree_vm_list_set_ref_retain(input_list.get(), i, &inputs[i]));
    }

    ref<iree_vm_list_t> output_list;
    IREE_RETURN_IF_ERROR(iree_vm_list_create(iree_vm_make_undefined_type_def(),
                                             8, iree_allocator_system(),
                                             &output_list));

    iree_vm_function_t function;
    IREE_RETURN_IF_ERROR(iree_vm_module_lookup_function_by_name(
        bytecode_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT,
        iree_make_cstring_view(function_name), &function));

    iree_vm_async_invoke_state_t state = {};

    iree_loop_t loop;
    iree_status_t loop_status;
    AllocateLoop(&loop_status, iree_allocator_system(), &loop);

    iree_status_t invoke_status;
    iree_vm_async_invoke(
        loop, &state, context_, function, IREE_VM_INVOCATION_FLAG_NONE,
        /*policy=*/NULL, input_list.get(), output_list.get(),
        iree_allocator_system(), async_callback, &invoke_status);

    IREE_RETURN_IF_ERROR(iree_loop_drain(loop, iree_infinite_timeout()));
    IREE_RETURN_IF_ERROR(invoke_status);

    std::vector<iree_vm_ref_t> outputs;
    outputs.resize(iree_vm_list_size(output_list.get()));
    for (iree_host_size_t i = 0; i < outputs.size(); ++i) {
      IREE_RETURN_IF_ERROR(
          iree_vm_list_get_ref_retain(output_list.get(), i, &outputs[i]));
    }
    return outputs;
  }

  iree_vm_instance_t* instance_ = nullptr;
  iree_vm_context_t* context_ = nullptr;
  iree_vm_module_t* bytecode_module_ = nullptr;
  iree_vm_module_t* async_runtime_module_ = nullptr;
  iree_vm_module_t* async_test_module_ = nullptr;
};

TEST_F(AsyncRuntimeModuleTest, FuncAwaitDelayedToken) {
  EXPECT_THAT(
      RunFunction("await_delayed_token", std::vector<iree_vm_value_t>()),
      IsOkAndHolds(Eq(MakeValuesList({42}))));
}

TEST_F(AsyncRuntimeModuleTest, FuncAwaitAvailableValue) {
  EXPECT_THAT(
      RunFunction("await_available_value", std::vector<iree_vm_value_t>()),
      IsOkAndHolds(Eq(MakeValuesList({84}))));
}

TEST_F(AsyncRuntimeModuleTest, FuncAwaitDelayedValue) {
  EXPECT_THAT(
      RunFunction("await_delayed_value", std::vector<iree_vm_value_t>()),
      IsOkAndHolds(Eq(MakeValuesList({84}))));
}

TEST_F(AsyncRuntimeModuleTest, FuncAwaitTokenError) {
  EXPECT_THAT(RunFunction("await_token_error", std::vector<iree_vm_value_t>()),
              StatusIs(StatusCode::kInternal));
}

TEST_F(AsyncRuntimeModuleTest, FuncAwaitDelayedMemref) {
  EXPECT_THAT(RunFunction("await_delayed_memref", std::vector<iree_vm_ref_t>()),
              IsOkAndHolds(Eq(MakeHalBufferViewList({"2xf32=1.0 2.0"}))));
}

}  // namespace
