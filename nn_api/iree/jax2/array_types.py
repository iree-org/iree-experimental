# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Sequence
import numpy as np
import numpy.lib.mixins

from . import ir_utils
from . import tracing

import jax.core

from iree.compiler import (
    ir,)

_BASE_HANDLED_FUNCTIONS = {}


def _base_implements(np_function):
  """Decorator that registers a base class implementation."""

  def decorator(func):
    _BASE_HANDLED_FUNCTIONS[np_function] = func
    return func

  return decorator


class TracedArrayBase(numpy.lib.mixins.NDArrayOperatorsMixin):
  """Base class for tracked arrays."""

  def __init__(self, aval: jax.core.AbstractValue):
    self.aval = aval

  def __array_function__(self, func, types, args, kwargs):
    if func not in _BASE_HANDLED_FUNCTIONS:
      return NotImplemented
    return _BASE_HANDLED_FUNCTIONS[func](*args, **kwargs)

  def __array__(self, dtype=None):
    assert dtype is None
    return self


@_base_implements(np.shape)
def _(arr: TracedArrayBase):
  return arr.aval.shape


@_base_implements(np.result_type)
def _(arr: TracedArrayBase):
  return arr.aval.dtype


class ExportedGlobalArray(TracedArrayBase, tracing.Intrinsic):
  """Represents an exported global exposed as one array at the Python level."""

  def __init__(self, aval: jax.core.ShapedArray, symbol_name: str,
               ir_type: ir.Type):
    super().__init__(aval)
    self.symbol_name = symbol_name
    self.ir_type = ir_type

  def __repr__(self):
    return f"ExportedGlobalArray(@{self.symbol_name} : {self.ir_type})"

  def resolve_ir_values(
      self, func_trace: tracing.FunctionIrTrace) -> Sequence[ir.Value]:
    return (ir_utils.create_global_load_op(self.symbol_name, self.ir_type),)


class IrValueArray(TracedArrayBase, tracing.Intrinsic):
  """Represents an array that corresponds to an IR value."""

  def __init__(self, aval: jax.core.ShapedArray, ir_value: ir.Value):
    super().__init__(aval)
    self.ir_value = ir_value

  def __repr__(self):
    return f"IrValueArray(@{self.ir_value})"

  def resolve_ir_values(
      self, func_trace: tracing.FunctionIrTrace) -> Sequence[ir.Value]:
    return (self.ir_value,)
