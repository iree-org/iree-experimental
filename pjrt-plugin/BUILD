# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cc_binary(
    name = "libIreePjrtPlugin.so",
    linkshared = 1,
    linkstatic = 1,
    srcs = [
        "entry_point.cc",
        "pjrt_plugin_exports.h",
    ],
    deps = [
        ":impl",
        "@iree//runtime/src/iree/base/internal:path",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt/c:pjrt_c_api_hdrs",
    ],
    defines = [
        "PJRT_PLUGIN_BUILDING_LIBRARY",
    ],
    linkopts = [
        "-Wl,--no-undefined",
    ],
)

cc_library(
    name = "impl",
    srcs = [
        "pjrt_plugin_impl.cc",
    ],
    hdrs = [
        "pjrt_plugin_impl.h",
    ],
    deps = [
        "@iree//runtime/src/iree/vm",
        "@iree//runtime/src/iree/hal",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)
