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

// Gets the version of the plugin API. This is incremented on breaking changes
// to the exported plugin API.
PJRT_PLUGIN_EXPORTED unsigned PJRT_Plugin_ApiVersion();
typedef unsigned PJRT_Plugin_ApiVersion_FN();

// Primary entry-point for the plugin, retrieving a concrete API struct.
// Since this is always a dynamic symbol loaded from a shared library (i.e.
// if static linking, a different mechanism would be used), we require the
// caller to inform us the path they used to load the library. This allows
// plugin libraries to exist as part of a relocatable directory tree that
// consists of additional files which must be peers in some defined way.
// If the plugin cannot be initialized, it must return nullptr.
PJRT_PLUGIN_EXPORTED PJRT_Api* PJRT_Plugin_Create(const char* plugin_path);
typedef PJRT_Api* PJRT_Plugin_Create_FN(const char* plugin_path);

// Destroys an API pointer previously obtained via PJRT_Plugin_Initialize.
// This may perform heavy-weight shutdown activities, depending on the
// implementation.
PJRT_PLUGIN_EXPORTED void PJRT_Plugin_Destroy(PJRT_Api* api);
typedef void PJRT_Plugin_Destroy_FN(PJRT_Api* api);

#ifdef __cplusplus
}
#endif

#endif  // IREE_PJRT_PLUGIN_PJRT_PLUGIN_EXPORTS_H_
