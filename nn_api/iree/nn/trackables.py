# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Represents Python entities bound to components of a program.

Trackables:
  - Can stand in symbolically for unknown Python values.
  - Emit IR for different kinds of access.
"""

from iree.compiler import ir
from iree.compiler.dialects import (
  iree_input as input_d,
)


class Trackable:
  ...


class GlobalTrackable(Trackable):
  """Trackable which mirrors one or more iree_input.global entities."""
  def __init__(self, *global_ops: input_d.GlobalOp):
    super().__init__()
    self.global_ops = global_ops

