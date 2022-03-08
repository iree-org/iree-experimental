# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from contextlib import contextmanager
import threading
from typing import Any, List, Sequence

from iree.compiler import (
    ir,)
from iree.compiler.dialects import (
    builtin as builtin_d,
    iree_input as iree_input_d,
    func as func_d,
)

import jax.core
from jax.tree_util import (tree_map, tree_flatten, tree_unflatten)
from numpy import number

_thread_state = threading.local()


class Intrinsic:
  """Objects which interact natively with the tracing system implement this."""

  def resolve_ir_values(self,
                        func_trace: "FunctionIrTrace") -> Sequence[ir.Value]:
    raise NotImplementedError(
        f"Cannot use {self} as an expression in an export function")

  def resolve_call(self, func_trace: "FunctionIrTrace", *args, **kwargs):
    raise NotImplementedError(
        f"Cannot use {self} as the target of a call in an export function")


class CallableIntrinsic(Intrinsic):
  """Intrinsic subclass that supports calls.

  This is separate so as to make error handling better (i.e. does not support
  calls) for intrinsics that are not callable.
  """

  def __call__(self, *args, **kwargs):
    return current_ir_trace().handle_call(self, args, kwargs)


class IrTrace:
  """Gets callbacks for tracing events."""

  def finalize(self):
    """Called when the trace is finished (popped off the stack)."""
    pass

  def handle_call(self, target: Intrinsic, args, kwargs):
    raise NotImplementedError(f"The current trace scope does not support calls")


class ImmediateIrTrace(IrTrace):
  ...


def _trace_scopes() -> List[IrTrace]:
  try:
    trace_scopes = _thread_state.trace_scopes
  except AttributeError:
    trace_scopes = _thread_state.trace_scopes = [ImmediateIrTrace()]
  return trace_scopes


@contextmanager
def new_ir_trace_scope(ir_trace: IrTrace):
  trace_scopes = _trace_scopes()
  trace_scopes.append(ir_trace)
  try:
    yield ir_trace
  finally:
    ir_trace.finalize()
    del trace_scopes[-1]


def current_ir_trace() -> IrTrace:
  return _trace_scopes()[-1]


class FunctionIrTrace(IrTrace):
  """Captures execution into a `func` body."""

  def __init__(self, *, func_op: builtin_d.FuncOp, module: ir.Module,
               module_symbol_table: ir.SymbolTable):
    self.func_op = func_op
    self.module = module
    self.module_symbol_table = module_symbol_table
    self.context = func_op.context
    self.ip = ir.InsertionPoint(self.func_op.entry_block)
    self.return_types = None
    self.loc = self.func_op.location

  @property
  def arguments(self) -> Sequence[ir.Value]:
    return list(self.func_op.entry_block.arguments)

  def handle_call(self, target: Intrinsic, args, kwargs):
    with self.loc, self.ip:
      return target.resolve_call(self, *args, **kwargs)

  def materialize_py_values(self, py_value: Any) -> Sequence[ir.Value]:
    # TODO: IR'izing values could probably use a dedicated helper.
    if isinstance(py_value, (list, tuple, dict)):
      # Treat it as a tree.
      py_flat, tree_def = tree_flatten(py_value)
      ir_flat = []
      for py_flat_item in py_flat:
        ir_flat.extend(self.materialize_py_values(py_flat_item))
      return ir_flat

    # Unwrap ConcreteArray.
    if isinstance(py_value, jax.core.ConcreteArray):
      py_value = py_value.val
    if isinstance(py_value, Intrinsic):
      with self.loc, self.ip:
        return py_value.resolve_ir_values(self)
    if isinstance(py_value, number):
      return py_value

    raise TypeError(
        f"While tracing, encountered an unsupported value: {py_value}")

  def emit_return(self, *ir_values: Sequence[ir.Value]):
    with self.loc, self.ip:
      func_d.ReturnOp(ir_values)
      # Check or rewrite the function return type.
      value_types = [v.type for v in ir_values]
      if self.return_types:
        if value_types != self.return_types:
          raise ValueError(f"Multi-return function must return same types. "
                           f"{value_types} vs {self.return_types}")
        return
      self.return_types = value_types
      ftype = self.func_op.type
      ftype = ir.FunctionType.get(ftype.inputs, value_types)
      self.func_op.attributes["type"] = ir.TypeAttr.get(ftype)
      assert self.func_op.verify(), "Created function is invalid"
