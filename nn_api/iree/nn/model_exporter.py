# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional
from iree.compiler import (
    ir,
)
from iree.compiler.dialects import (
    arith as arith_d,
    builtin as builtin_d,
    chlo as chlo_d,
    iree_input as iree_input_d,
    linalg as linalg_d,
    mhlo as mhlo_d,
    std as std_d,
    tensor as tensor_d,
)


class ModelExporter:
  """Exports an NN model as an IREE compilable program."""

  def __init__(self, *, context: Optional[ir.Context], module=None):
    if not context:
      context = ir.create_context()
    self.context = context
    with context:
      self.module = module or self._impl.builtin_d.ModuleOp.create(
          self._impl.ir.Location.unknown.get())

  def create_global_op(self, name: str, *, mutable: bool = True, initial_value = None):
    ...

  def add_linear_parameters(self):
    """Adds a linear list of parameters to the model.

    Linear parameters are always accessed as a group and have one logical
    name within the model. Further interpretation of the group of parameters
    may be imposed through additional metadata (i.e. accessing as a tree, etc).
    """
    ...
