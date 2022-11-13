// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include <cstring>

#include "iree/base/internal/path.h"  // TODO: Make public trampoline.
#include "iree/vm/api.h"
#include "pjrt_plugin_exports.h"
#include "pjrt_plugin_impl.h"

//===----------------------------------------------------------------------===//
// Exports from the shared library
//===----------------------------------------------------------------------===//

unsigned PJRT_Plugin_ApiVersion() { return 1; }

PJRT_Api* PJRT_Plugin_Create(const char* plugin_path) {
  iree_string_view_t lib_dir_sv = iree_file_path_dirname(
      iree_make_string_view(plugin_path, std::strlen(plugin_path)));
  fprintf(stderr, "IREE PJRT_Plugin_Initialize: lib_dir=%*s\n",
          (int)lib_dir_sv.size, lib_dir_sv.data);
  PJRT_Api* api = new PJRT_Api;
  std::memset(api, 0, sizeof(*api));
  if (!iree::pjrt::Initialize(
          api, "vulkan",  // TODO: Derive
          std::string_view(lib_dir_sv.data, lib_dir_sv.size))) {
    delete api;
    return nullptr;
  }
  return api;
}

void PJRT_Plugin_Destroy(PJRT_Api* api) {
  iree::pjrt::Deinitialize(api);
  delete api;
}
