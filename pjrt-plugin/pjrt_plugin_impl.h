// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// PJRT presently does not define how to load a library and get the PJRT_Api
// struct (the one in-tree use of this statically loads the TPU support
// library). This header posits a minimal export API intended to be made
// available as public symbols on a built shared library.

#ifndef IREE_PJRT_PLUGIN_PJRT_PLUGIN_IMPL_H_
#define IREE_PJRT_PLUGIN_PJRT_PLUGIN_IMPL_H_

#include <string>
#include <string_view>
#include <unordered_map>

#include "pjrt_plugin_defs.h"
#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"

namespace iree::pjrt {

// Convenience C++ wrapper around logging callbacks.
class Logger {
 public:
  Logger(PJRT_LogCallbacks callbacks) : callbacks_(callbacks) {}

  void error(std::string_view message) {
    callbacks_.sink(callbacks_.context, PJRT_ERROR, message.data(),
                    message.size());
  }

 private:
  PJRT_LogCallbacks callbacks_;
};

// Manages configuration variables.
struct ConfigVars {
  ConfigVars() = default;
  bool Parse(Logger& logger, const char** config_vars, size_t config_var_size);

  std::string driver_name = "vulkan";
};

// Constructs and initializes a new API struct based on a given location for
// shared libraries and tools. In the normal flow where this is initialized
// from a shared library, this will be the directory name containing the plugin
// library. For static linking cases, it must be arrived at by other means.
//
// IREE uses the library directory to locate compiler tools, binaries and
// other support files. If it is empty, a default heuristic is used.
// Returns false on failure.
bool Initialize(PJRT_Api* api, Logger& logger, ConfigVars config_vars);

// Deinitializes an API pointer populated via Initialize. This may perform
// heavy-weight shutdown activities.
void Deinitialize(PJRT_Api* api);

}  // namespace iree::pjrt

#endif  // IREE_PJRT_PLUGIN_PJRT_PLUGIN_IMPL_H_
