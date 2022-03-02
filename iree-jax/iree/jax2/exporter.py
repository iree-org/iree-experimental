# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Low level facilities for exporting JAX programs.

This is not the primary user-oriented API (although pieces of types here will
be user manipulable in the course of interacting with the machinery).
"""

from collections import namedtuple
import functools
import logging
from typing import Any, Dict, Optional, Sequence
import weakref

import numpy as np
import jax.core
from numpy import array

from . import array_types
from . import ir_utils
from . import jax_utils
from . import tracing

from iree.compiler import (
    ir,)

from jax.tree_util import (tree_map, tree_flatten, tree_unflatten)

__all__ = [
    "ExportModule",
]

logger = logging.getLogger("iree_jax")

# Opaque value to indicate something is empty. Used in cases where 'None'
# may have a different meaning.
_Empty = object()


class ExportModule:
  """An exported module under construction."""

  def __init__(self, *, module: ir.Module):
    self.module = module
    self.context = module.context
    self.op = self.module.operation
    self.loc = self.op.location
    self._body = self.op.regions[0].blocks[0]
    self._symbol_table = ir.SymbolTable(self.op)
    self.ip = ir.InsertionPoint(self._body)
    self._refs = RefTracker()
    self.exports = AttributeDict()

  @classmethod
  def create_empty(cls,
                   *,
                   context: Optional[ir.Context] = None,
                   name: Optional[str] = None):
    if not context:
      context = ir_utils.create_context()
    loc = ir.Location.unknown(context=context)
    module = ir.Module.create(loc)
    if name:
      module.operation.attributes["sym_name"] = ir.StringAttr.get(
          name, context=context)
    return cls(module=module)

  def def_global(self,
                 symbol_name: str,
                 value,
                 *,
                 initialize: bool = False,
                 mutable: bool = True) -> jax.core.AbstractValue:
    if symbol_name in self.exports:
      raise ValueError(f"Duplicate export definition: {symbol_name}")
    # Abstractify.
    concrete_value = _Empty
    if not isinstance(value, jax.core.AbstractValue):
      concrete_value = value
      value = jax_utils.abstractify(concrete_value)

    # Convert to IR type.
    ir_types = jax_utils.aval_to_ir_types(self.context, value)

    # If we have a concrete value, then it may already be associated with a
    # global.
    info = self._refs.track(concrete_value)
    if info.tracked_value is not _Empty:
      return info.tracked_value

    # TODO: Other types?
    result = None
    if isinstance(value, jax.core.ShapedArray):
      if len(ir_types) != 1:
        raise TypeError(f"Composite JAX types not yet supported: {ir_types}")
      with self.loc, self.ip:
        initial_value = None
        if initialize:
          initial_value = ir_utils.create_array_attribute(
              concrete_value, ir_types)
        actual_symbol_name = ir_utils.create_global(self._symbol_table,
                                                    symbol_name,
                                                    ir_types[0],
                                                    mutable=mutable,
                                                    initial_value=initial_value)
      info.tracked_value = array_types.ExportedGlobalArray(
          value, actual_symbol_name, ir_types[0])
      result = jax.core.ConcreteArray(value.dtype, info.tracked_value)
    else:
      raise TypeError(f"Export not implemented for JAX abstract value: {value}")

    self.exports[symbol_name] = result
    return result

  def def_global_tree(self,
                      symbol_name: str,
                      treeish: Any,
                      *,
                      initialize: bool = False,
                      mutable: bool = True) -> Any:
    """Defines a PyTree with a symbolic name.

    Note that the tree can either consist of concrete or abstract values.
    If concrete, only their metadata is retained (the backing values will not
    have initializers set).
    """
    if symbol_name in self.exports:
      raise ValueError(f"Duplicate export definition: {symbol_name}")

    concrete_leaves, tree_def = tree_flatten(treeish)
    imported_leaves = []
    tracked_leaf_count = 0
    for concrete_leaf in concrete_leaves:
      # We fork between trackable things and static constants. Currently this
      # is just array vs not, but this should match Jax's heuristic.
      # TODO: Make sure this is the right way to detect array.
      if hasattr(concrete_leaf, "__array__"):
        leaf_symbol = f"{symbol_name}${tracked_leaf_count}"
        logger.debug("def_global_tree: array %s=%r:%r", leaf_symbol,
                     concrete_leaf.shape, concrete_leaf.dtype)
        imported_leaves.append(
            self.def_global(leaf_symbol,
                            concrete_leaf,
                            initialize=initialize,
                            mutable=mutable))
        tracked_leaf_count += 1
      else:
        logger.debug("def_global_tree: literal=%r", type(concrete_leaf))
        imported_leaves.append(concrete_leaf)
    result = tree_unflatten(tree_def, imported_leaves)
    logger.debug("def_global_tree: new tree=%r", result)
    self.exports[symbol_name] = result
    return result

  def def_func(self,
               f=None,
               *,
               symbol_name: Optional[str] = None,
               arguments: Sequence[Any] = ()):
    if f is None:
      return functools.partial(self.def_func,
                               symbol_name=symbol_name,
                               arguments=arguments)
    if symbol_name is None:
      # TODO: Inspect name properly.
      symbol_name = f.__name__
    if symbol_name in self.exports:
      raise ValueError(f"Duplicate export definition: {symbol_name}")

    # Unpack arguments.
    # TODO: Make this match how JAX handles arguments (and make nicer).
    arguments_flat, arguments_tree_def = tree_flatten(arguments)
    argument_avals = [jax_utils.abstractify(x) for x in arguments_flat]
    argument_ir_types = []
    for argument_aval in argument_avals:
      aval_ir_types = jax_utils.aval_to_ir_types(self.context, argument_aval)
      if len(aval_ir_types) != 1:
        raise TypeError(f"TODO: Composite abstract types not yet supported")
      argument_ir_types.extend(aval_ir_types)

    with self.loc, self.ip:
      actual_symbol_name, func_op = ir_utils.create_func_op(
          self._symbol_table, symbol_name, argument_ir_types)

    with tracing.new_ir_trace_scope(
        tracing.FunctionIrTrace(
            func_op=func_op,
            module=self.module,
            module_symbol_table=self._symbol_table)) as trace:
      # TODO: Lots wrong with the argument packing/unpacking here. Should
      # map trees in the same way JAX does.
      argument_py_values = [
          array_types.IrValueArray(aval, ir_value)
          for aval, ir_value in zip(argument_avals, trace.arguments)
      ]
      argument_py_tree = tree_unflatten(arguments_tree_def, argument_py_values)
      return_py_value = f(*argument_py_tree)
      if return_py_value is None:
        trace.emit_return()
      else:
        return_ir_values = trace.materialize_py_values(return_py_value)
        trace.emit_return(*return_ir_values)

  def __str__(self):
    return str(self.module)


class RefInfo:
  __slots__ = [
      "_referrent",
      "tracked_value",
  ]

  def __init__(self, referrent: Any):
    self._referrent = weakref.ref(referrent)
    self.tracked_value = _Empty


class RefTracker:
  """Tracks live references from Python values to symbolic associations."""

  def __init__(self):
    self._refs: Dict[int, RefInfo] = {}

  def track(self, referrent: Any) -> RefInfo:
    ref_id = id(referrent)
    existing = self._refs.get(ref_id)
    if existing:
      return existing
    info = RefInfo(referrent)
    weakref.finalize(referrent, self._ref_finalizer, ref_id)
    self._refs[ref_id] = info
    return info

  def _ref_finalizer(self, ref_id: int):
    del self._refs[ref_id]


class AttributeDict(dict):
  """Dict that allows access to keys as attributes.

  TODO: Move this to the higher level user API when it exists.
  """

  def __getattribute__(self, name: str) -> Any:
    try:
      return self[name]
    except KeyError:
      raise AttributeError(name)
