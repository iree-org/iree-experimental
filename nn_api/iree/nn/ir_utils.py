# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import contextlib
from collections import namedtuple

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from iree.compiler import (
    ir,
)
from iree.compiler.api import (
    driver,
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

# Represents a value imported into the system, allowing a separation between
# a closed type system (imported) that overlays a larger Python type system.
ImportedValue = namedtuple("ImportedValue", "imported,original")

# Type aliases.
TypeMapper = Callable[["ImportContext", ImportedValue], Tuple[ir.Type]]


class ImportHooks:
  """Hooks for extending import."""

  def __init__(self):
    # Type mapping.
    self._type_mappers_by_type: Dict[type, Union[bool, TypeMapper]] = dict()

  def preprocess_module(self, ic: "ImportContext", module: ir.Module):
    """Performs preprocessing cleanup tasks on a module."""
    pass

  def import_python_value(self, py_value: Any) -> ImportedValue:
    """Imports an arbitrary Python value.

    By default, this returns an identity.
    """
    return ImportedValue(py_value, py_value)

  def add_type_mapper(self, type_match: type, mapper: TypeMapper):
    """Adds a type mapper which represents the given imported type."""
    self._type_mappers_by_type[type_match] = mapper

  def lookup_type_mapper(self, value: ImportedValue) -> TypeMapper:
    """Looks up a TypeMapper for the given Python value.

    Raises:
      TypeError on no mapping.
    """
    match_value = value.imported
    mapper = self._type_mappers_by_type.get(match_value.__class__)
    if mapper:
      return mapper
    if mapper != False:
      # Scan for a match.
      for match_type, mapper in self._type_mappers_by_type.items():
        if isinstance(match_value, match_type):
          # Sub-type match.
          self._type_mappers_by_type[match_value.__class__] = mapper
          return mapper
      else:
        # Tombstone.
        self._type_mappers_by_type[match_value.__class__] = False

    raise TypeError(f"Could not find type mapper for class {value.__class__}"
                    f" ({value})")

  def map_value(self, ic: "ImportContext", value: Any) -> Tuple[ir.Type]:
    """Maps a Python value to a concrete sequence of IR types."""
    type_mapper = self.lookup_type_mapper(value)
    return tuple(type_mapper(ic, value))


class ImportContext:
  """Context for importing Python structures into IR."""

  def __init__(self,
               *,
               hooks: Optional[ImportHooks] = None,
               context: Optional[ir.Context] = None,
               module: Optional[builtin_d.ModuleOp] = None):
    self.hooks = hooks or self.create_hooks()
    self.context = context if context else self.create_context()
    self.loc = ir.Location.unknown(context=self.context)
    self._root_module: Optional[ir.Module] = None
    if module:
      self.module = module
    else:
      self._root_module = ir.Module.create(self.loc)
      self.module = self._root_module.operation
    # TODO: Add a "body" attribute to builtin.module.
    self.module_body = self.module.regions[0].blocks[0]
    self.symbol_table = ir.SymbolTable(self.module)
    self._ip_stack = []

  def __str__(self):
    if self._root_module:
      return str(self._root_module)
    else:
      return str(self.module)

  def set_file_line_col(self, file: str, line: int, col: int):
    self.loc = ir.Location.file(file, line, col, context=self.context)

  @contextlib.contextmanager
  def scoped_push_ip(self, scoped_ip: ir.InsertionPoint):
    self.push_ip(scoped_ip)
    try:
      yield scoped_ip
    finally:
      self.pop_ip()

  def push_ip(self, scoped_ip: ir.InsertionPoint):
    self._ip_stack.append(scoped_ip)

  def pop_ip(self):
    assert self._ip_stack, "Mismatched push_ip/pop_ip: stack is empty on pop"
    del self._ip_stack[-1]

  @property
  def ip(self) -> ir.InsertionPoint:
    assert self._ip_stack, "InsertionPoint requested but stack is empty"
    return self._ip_stack[-1]

  def create_hooks(self) -> ImportHooks:
    return ImportHooks()

  def create_context(self, *, debug: bool = False) -> ir.Context:
    context = ir.Context()
    if debug:
      context.enable_multithreading(False)
    iree_input_d.register_dialect(context)
    return context
