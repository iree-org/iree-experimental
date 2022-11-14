// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include <cstring>

#include "iree/base/internal/path.h"  // TODO: Make public trampoline.
#include "iree/vm/api.h"
#include "pjrt_plugin_defs.h"
#include "pjrt_plugin_impl.h"

//===----------------------------------------------------------------------===//
// Declarations (needed to set visibility and import/export flags).
//===----------------------------------------------------------------------===//

PJRT_PLUGIN_EXPORTED unsigned PJRT_Plugin_ApiVersion();
static_assert(std::is_same_v<decltype(PJRT_Plugin_ApiVersion),
                             PJRT_Plugin_ApiVersion_FN>);

PJRT_PLUGIN_EXPORTED PJRT_Api* PJRT_Plugin_Create(
    PJRT_LogCallbacks log_callbacks, const char** config_vars,
    size_t config_var_size);
static_assert(
    std::is_same_v<decltype(PJRT_Plugin_Create), PJRT_Plugin_Create_FN>);

PJRT_PLUGIN_EXPORTED void PJRT_Plugin_Destroy(PJRT_Api* api);
static_assert(
    std::is_same_v<decltype(PJRT_Plugin_Destroy), PJRT_Plugin_Destroy_FN>);

//===----------------------------------------------------------------------===//
// Exports from the shared library
//===----------------------------------------------------------------------===//

unsigned PJRT_Plugin_ApiVersion() { return 1; }

PJRT_Api* PJRT_Plugin_Create(PJRT_LogCallbacks log_callbacks,
                             const char** config_vars, size_t config_var_size) {
  iree::pjrt::Logger logger(log_callbacks);
  iree::pjrt::ConfigVars parsed_config_vars;
  if (!parsed_config_vars.Parse(logger, config_vars, config_var_size)) {
    return nullptr;
  }

  PJRT_Api* api = new PJRT_Api;
  std::memset(api, 0, sizeof(*api));
  if (!iree::pjrt::Initialize(api, logger, parsed_config_vars)) {
    delete api;
    return nullptr;
  }
  return api;
}

void PJRT_Plugin_Destroy(PJRT_Api* api) {
  iree::pjrt::Deinitialize(api);
  delete api;
}
