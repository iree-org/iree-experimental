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
        #"pjrt_plugin_impl.cc",
    ],
    hdrs = [
        "api_impl.h",
        "dylib_entry_point.cc.inc",
        # "pjrt_plugin_defs.h",
        # "pjrt_plugin_impl.h",
    ],
    deps = [
        "@iree//runtime/src/iree/vm",
        "@iree//runtime/src/iree/hal",
        "@org_tensorflow//tensorflow/compiler/xla/pjrt/c:pjrt_c_api_hdrs",
    ],
)
