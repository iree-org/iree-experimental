// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// PJRT presently does not define how to load a library and get the PJRT_Api
// struct (the one in-tree use of this statically loads the TPU support
// library). This header posits a minimal export API intended to be made
// available as public symbols on a built shared library.

//===----------------------------------------------------------------------===//
// Visibility annotations.
// Use PJRT_PLUGIN_EXPORTED for exported functions.
//
// On Windows, if PJRT_PLUGIN_ENABLE_WINDOWS_DLL_DECLSPEC is defined, then
// __declspec(dllexport) and __declspec(dllimport) will be generated. This
// can only be enabled if actually building DLLs. It is generally, mutually
// exclusive with the use of other mechanisms for managing imports/exports
// (i.e. CMake's WINDOWS_EXPORT_ALL_SYMBOLS feature).
//===----------------------------------------------------------------------===//

#ifndef IREE_PJRT_PLUGIN_PJRT_PLUGIN_EXPORTS_H_
#define IREE_PJRT_PLUGIN_PJRT_PLUGIN_EXPORTS_H_

#include "tensorflow/compiler/xla/pjrt/c/pjrt_c_api.h"

#if (defined(_WIN32) || defined(__CYGWIN__)) && \
    !defined(PJRT_PLUGIN_ENABLE_WINDOWS_DLL_DECLSPEC)
// Visibility annotations disabled.
#define PJRT_PLUGIN_EXPORTED
#elif defined(_WIN32) || defined(__CYGWIN__)
// Windows visibility declarations.
#if PJRT_PLUGIN_BUILDING_LIBRARY
#define PJRT_PLUGIN_EXPORTED __declspec(dllexport)
#else
#define PJRT_PLUGIN_EXPORTED __declspec(dllimport)
#endif
#else
// Non-windows: use visibility attributes.
#if PJRT_PLUGIN_BUILDING_LIBRARY
#define PJRT_PLUGIN_EXPORTED __attribute__((visibility("default")))
#else
#define PJRT_PLUGIN_EXPORTED
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

// Logging is done via a level and callbacks to determine if enabled and sink
// a message.
typedef enum {
  PJRT_VERBOSE = -1,
  PJRT_INFO = 0,
  PJRT_WARNING = 1,
  PJRT_ERROR = 2,
  PJRT_FATAL = 3,
} PJRT_LogLevel;

typedef bool PJRT_LogEnabled_FN(void* context, PJRT_LogLevel level);
typedef void PJRT_LogSink_FN(void* context, PJRT_LogLevel level,
                             const char* message, size_t message_size);

typedef struct {
  void* context;
  PJRT_LogEnabled_FN* enabled;
  PJRT_LogSink_FN* sink;
} PJRT_LogCallbacks;

// Gets the version of the plugin API. This is incremented on breaking changes
// to the exported plugin API.
// Exported as: PJRT_Plugin_ApiVersion
typedef unsigned PJRT_Plugin_ApiVersion_FN();

// Primary entry point for creating a specific PJRT API instance, configuring
// it with logging callbacks and configuration variables. Configuration
// variables are represented as "VAR=VALUE" simular to environment variables.
// It is recommended (and up to the implementation) to also have correspondance
// with environment variables.
// Exported as: PJRT_Plugin_Create
typedef PJRT_Api* PJRT_Plugin_Create_FN(PJRT_LogCallbacks log_callbacks,
                                        const char** config_vars,
                                        size_t config_var_size);

// Destroys an API pointer previously obtained via PJRT_Plugin_Initialize.
// This may perform heavy-weight shutdown activities, depending on the
// implementation.
// Exported as PJRT_Plugin_Destroy
typedef void PJRT_Plugin_Destroy_FN(PJRT_Api* api);

#ifdef __cplusplus
}
#endif

#endif  // IREE_PJRT_PLUGIN_PJRT_PLUGIN_EXPORTS_H_
