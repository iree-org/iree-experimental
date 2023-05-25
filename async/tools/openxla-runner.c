// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode/module.h"
#include "openxla/runtime/async/loop_async.h"
#include "openxla/runtime/async/module.h"

void FreeLoop(iree_allocator_t allocator, iree_loop_t loop);

iree_status_t async_callback(void* user_data, iree_loop_t loop,
                             iree_status_t status, iree_vm_list_t* outputs) {
  if (iree_status_is_ok(status)) {
    fprintf(stdout, "Async invocation finished\n");
    fflush(stdout);
  }

  iree_vm_list_release(outputs);
  return iree_ok_status();
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr,
            "Usage:\n"
            "  openxla-runner - <entry.point> # read from stdin\n"
            "  openxla-runner </path/to/say_hello.vmfb> <entry.point>\n");
    fprintf(stderr, "  (See the README for this sample for details)\n ");
    return -1;
  }

  iree_allocator_t allocator = iree_allocator_system();

  iree_vm_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                        allocator, &instance));
  IREE_CHECK_OK(openxla_async_runtime_module_register_types(instance));
  iree_vm_module_t* async_runtime_module = NULL;
  IREE_CHECK_OK(iree_async_runtime_module_create(instance, allocator,
                                                 &async_runtime_module));

  const char* module_path = argv[1];
  iree_file_contents_t* module_contents = NULL;
  if (strcmp(module_path, "-") == 0) {
    IREE_CHECK_OK(iree_stdin_read_contents(allocator, &module_contents));
  } else {
    IREE_CHECK_OK(
        iree_file_read_contents(module_path, allocator, &module_contents));
  }

  iree_vm_module_t* bytecode_module = NULL;
  IREE_CHECK_OK(iree_vm_bytecode_module_create(
      instance, module_contents->const_buffer,
      iree_file_contents_deallocator(module_contents), allocator,
      &bytecode_module));

  iree_vm_module_t* modules[] = {async_runtime_module, bytecode_module};
  iree_vm_context_t* context = NULL;
  IREE_CHECK_OK(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE, IREE_ARRAYSIZE(modules), modules,
      allocator, &context));

  iree_vm_function_t function;
  IREE_CHECK_OK(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(argv[2]), &function));

  fprintf(stdout, "INVOKE BEGIN %s\n", argv[2]);
  fflush(stdout);
  iree_vm_list_t* outputs = NULL;

  iree_vm_async_invoke_state_t state = {0};
  iree_status_t status = iree_vm_list_create(
      /*element_type=*/iree_vm_make_undefined_type_def(),
      /*capacity=*/1, allocator, &outputs);

  if (!iree_status_is_ok(status)) {
    fprintf(stdout, "can't allocate output vm list");
    fflush(stdout);
    return 0;
  }

  iree_status_t loop_status = iree_ok_status();
  iree_loop_t loop = iree_loop_async(&loop_status);

  iree_vm_async_invoke(loop, &state, context, function,
                       IREE_VM_INVOCATION_FLAG_NONE, /*policy=*/NULL,
                       /*inputs=*/NULL, outputs, allocator, async_callback,
                       &state);
  iree_vm_value_t result;
  iree_vm_list_get_value(outputs, 0, &result);
  fprintf(stdout, "Result is %d\n", result.i32);
  fprintf(stdout, "INVOKE END\n");
  fflush(stdout);

  iree_vm_list_release(outputs);
  outputs = NULL;
  iree_vm_context_release(context);
  iree_vm_module_release(bytecode_module);
  iree_vm_module_release(async_runtime_module);
  iree_vm_instance_release(instance);
  return 0;
}
