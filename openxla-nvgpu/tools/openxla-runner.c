// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/modules/hal/types.h"
#include "iree/runtime/api.h"
#include "openxla/runtime/nvgpu/cudnn_module.h"

// TODO: This is a temporary work around missing custom modules integration into
// IREE tools (iree-run-module). We already have flags to enable plugins in
// compiler tools (`iree-compiler` and `iree-opt`), but not yet in "runtime"
// tools. This tool can only run VM function with empty arguments and empty
// results, and intended for testing cuDNN custom module.
int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(stderr,
            "Usage:\n"
            "  openxla-runner - <entry.point> # read from stdin\n"
            "  openxla-runner </path/to/say_hello.vmfb> "
            "<entry.point>\n");
    return -1;
  }

  // Internally IREE does not (in general) use malloc and instead uses the
  // provided allocator to allocate and free memory. Applications can integrate
  // their own allocator as-needed.
  iree_allocator_t host_allocator = iree_allocator_system();

  // Create and configure the instance shared across all sessions.
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  IREE_CHECK_OK(iree_runtime_instance_create(&instance_options, host_allocator,
                                             &instance));

  // Try to create the CUDA device.
  iree_hal_device_t* device = NULL;
  IREE_CHECK_OK(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view("cuda"), &device));

  // Create one session per loaded module to hold the module state.
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  IREE_CHECK_OK(iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), &session));

  // Create the custom module that can be reused across contexts.
  iree_vm_module_t* custom_module = NULL;
  IREE_CHECK_OK(iree_custom_module_cudnn_create(
      iree_runtime_instance_vm_instance(instance), device, host_allocator,
      &custom_module));
  IREE_CHECK_OK(iree_runtime_session_append_module(session, custom_module));
  iree_vm_module_release(custom_module);

  // Load the module from stdin or a file on disk.
  const char* module_path = argv[1];
  if (strcmp(module_path, "-") == 0) {
    IREE_CHECK_OK(
        iree_runtime_session_append_bytecode_module_from_stdin(session));
  } else {
    IREE_CHECK_OK(iree_runtime_session_append_bytecode_module_from_file(
        session, module_path));
  }

  iree_string_view_t entry_point = iree_make_cstring_view(argv[2]);
  fprintf(stdout, "INVOKE BEGIN %.*s\n", (int)entry_point.size,
          entry_point.data);
  fflush(stdout);

  iree_vm_list_t* inputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(NULL, 1, host_allocator, &inputs));
  iree_vm_list_t* outputs = NULL;
  IREE_CHECK_OK(iree_vm_list_create(NULL, 1, host_allocator, &outputs));

  // Synchronously invoke the requested function.
  IREE_CHECK_OK(
      iree_runtime_session_call_by_name(session, entry_point, inputs, outputs));

  fprintf(stdout, "INVOKE END %.*s\n", (int)entry_point.size, entry_point.data);
  fflush(stdout);

  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);

  iree_runtime_session_release(session);
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);

  return 0;
}
