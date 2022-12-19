# Copyright 2022 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

cc_library(
    name = "client",
    srcs = [
      "client.cc",
    ],
    hdrs = [
      "client.h",
    ],
    deps = [
        "//iree/integrations/pjrt/common:impl",
        "@iree//runtime/src/iree/hal/local",
        "@iree//runtime/src/iree/hal/local:executable_loader",
        "@iree//runtime/src/iree/hal/local/loaders/registration",
        "@iree//runtime/src/iree/hal/drivers/local_task:task_driver",
        "@iree//runtime/src/iree/task:api",
    ],
)

cc_binary(
    name = "lib_pjrt_plugin_iree_cpu.so",
    linkshared = 1,
    linkstatic = 1,
    srcs = [
        "dylib_entry_point.cc",
    ],
    deps = [
        ":client",
        "//iree/integrations/pjrt/common:impl",
        "@iree//runtime/src/iree/base/internal:path",
    ],
    defines = [
        "PJRT_PLUGIN_BUILDING_LIBRARY",
    ],
    linkopts = [
        "-Wl,--no-undefined",
    ],
)
