# Copyright 2023 The OpenXLA Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Lit config for OpenXLA."""

# Lint for undefined variables is disabled as config is not defined inside this
# file, instead config is injected by way of evaluating runlit.cfg.py from
# runlit.site.cfg.py which in turn is evaluated by lit.py.
# pylint: disable=undefined-variable

import os
import tempfile

import lit.formats

config.name = "OpenXLA"
config.suffixes = [".mlir"]
config.test_format = lit.formats.ShTest(execute_external=True)

passthrough_env_vars = []
config.environment.update({
    k: v
    for k, v in os.environ.items()
    if k.startswith("IREE_") or k in passthrough_env_vars
})

# Use the most preferred temp directory.
config.test_exec_root = (os.environ.get("TEST_UNDECLARED_OUTPUTS_DIR") or
                         os.environ.get("TEST_TMPDIR") or
                         os.path.join(tempfile.gettempdir(), "lit"))
