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
    ],
    hdrs = [
        "api_impl.h",
        "dylib_entry_point.cc.inc",
    ],
    deps = [
        ":compiler",
        "@iree//runtime/src/iree/vm",
        "@iree//runtime/src/iree/vm:bytecode_module",
        "@iree//runtime/src/iree/hal",
        "@iree//runtime/src/iree/modules/hal",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt/c:pjrt_c_api_hdrs",
        "@org_tensorflow//tensorflow/compiler/xla:shape_util",
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
        "@iree//compiler/src/iree/compiler/API2/Stub",
    ],
)
