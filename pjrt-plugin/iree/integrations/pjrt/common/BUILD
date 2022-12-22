# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

package(
    default_visibility = ["//visibility:public"],
    features = ["layering_check"],
    licenses = ["notice"],  # Apache 2.0
)

cc_library(
    name = "impl",
    srcs = [
        "api_impl.cc",
        "platform.cc",
    ],
    hdrs = [
        "api_impl.h",
        "dylib_entry_point.cc.inc",
        "platform.h",
    ],
    deps = [
        ":compiler",
        "@iree//runtime/src/iree/hal",
        "@iree//runtime/src/iree/modules/hal",
        "@iree//runtime/src/iree/vm",
        "@iree//runtime/src/iree/vm:bytecode_module",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)

cc_library(
    name = "dylib_platform",
    srcs = [
        "dylib_platform.cc",
    ],
    hdrs = [
        "dylib_platform.h",
    ],
    deps = [
        ":impl",
        "@iree//compiler/src/iree/compiler/API2/Stub",
        "@iree//runtime/src/iree/base/internal:file_io",
        "@iree//runtime/src/iree/base/internal:path",
    ],
)

cc_library(
    name = "compiler",
    srcs = [
        "compiler.cc",
    ],
    hdrs = [
        "compiler.h",
    ],
    deps = [
        "@iree//compiler/src/iree/compiler/API2:EmbedHeaders",
    ],
)
