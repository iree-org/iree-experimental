# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_bytecode_module(
    NAME
      module_test_module
    SRC
      "module_test.mlir"
    C_IDENTIFIER
      "openxla_async_module_test_module"
    FLAGS
      "--compile-mode=vm"
    TESTONLY
    PUBLIC
)
