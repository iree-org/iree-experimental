# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler import ir
from iree.nn.jax_utils import (
    JaxImportContext,)
from iree.nn.module_builder import (
    ModuleBuilder,)


def run(f):
  print(f"BEGIN: {f.__name__}")
  context = ir.Context()
  with context:
    f()
  print(f"END: {f.__name__}\n")
  return f


@run
def basic():
  imp = ModuleBuilder(JaxImportContext())
  print(imp)
