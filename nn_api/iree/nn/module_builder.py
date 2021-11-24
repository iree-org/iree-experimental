# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from . import ir_utils

from iree.compiler import (
    ir,)
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


@dataclass(frozen=True)
class ImportedGlobal:
  """Encapsulates an imported global.

  An imported global is an association between an original Python value and
  the sequence of global ops that it expands to. It is identified by a unique
  logical identifier.
  """
  identifier: str
  py_value: Any
  component_symbols: Tuple[ir.StringAttr]
  component_ops: Tuple[iree_input_d.GlobalOp]

  @property
  def types(self) -> Sequence[ir.Type]:
    types: List[ir.Type] = []
    for op in self.component_ops:
      type_attr = ir.TypeAttr(op.attributes["type"])
      types.append(type_attr.value)
    return types


@dataclass(frozen=True)
class ImportedGlobalGroup:
  """Encapsulates a linearized sequence of globals which are used as a group."""
  identifier: str
  linear_py_values: Tuple[Any]
  globals: Tuple[ImportedGlobal]

  @property
  def flattened_symbols(self) -> Sequence[ir.StringAttr]:
    symbols = []
    for g in self.globals:
      symbols.extend(g.component_symbols)
    return symbols

  @property
  def flattened_types(self) -> Sequence[ir.Type]:
    types = []
    for g in self.globals:
      types.extend(g.types)
    return types

  @property
  def flattened_symbol_refs(self) -> Sequence[ir.FlatSymbolRefAttr]:
    return [
        ir.FlatSymbolRefAttr.get(s.value, context=s.context)
        for s in self.flattened_symbols
    ]


class ModuleBuilder:
  """Helpers for constructing an IREE input module."""

  def __init__(self, ic: Optional[ir_utils.ImportContext] = None):
    self.ic = ic if ic else ir_utils.ImportContext()
    self.ic.push_ip(ir.InsertionPoint(self.ic.module_body))
    self.globals: Dict[str, ImportedGlobal] = dict()
    self.global_groups: Dict[str, ImportedGlobalGroup] = dict()

  def __str__(self):
    return str(self.ic)

  def import_function(self,
                      module: Union[str, ir.Module],
                      main_symbol: str,
                      visibility: str = "private") -> str:
    """Imports a named function from another module into this one.

    Returns (imported symbol name, operation) of the found function (if
    present).

    TODO: This is horrible. Burn it.
    """
    ic = self.ic
    source_module = self.load_module(module)
    ic.hooks.preprocess_module(ic, source_module)

    with ic.context:
      target_module = ic.module
      target_symbol_table = ic.symbol_table
      target_body = target_module.regions[0].blocks[0]
      main_symbol_attr = ir.StringAttr.get(main_symbol)
      found_function = None
      found_name = None
      for source_operation in source_module.body.operations:
        source_operation = source_operation.detach_from_parent()
        target_body.append(source_operation)
        # TODO: Really should be checking for the Symbol trait.
        # TODO: The builtin.func overrides provide a 'name' attribute which
        # shadows the operation name.
        found_it = False
        if "sym_name" in source_operation.attributes:
          if source_operation.attributes["sym_name"] == main_symbol_attr:
            found_it = True
        target_symbol_table.insert(source_operation)
        if found_it:
          found_name = ir.StringAttr(
              source_operation.attributes["sym_name"]).value
          found_function = source_operation
          found_function.attributes["sym_visibility"] = ir.StringAttr.get(
              visibility)
    assert found_name, f"Imported function {main_symbol} not found"
    return found_name

  def load_module(self, module: Union[str, ir.Module]):
    if isinstance(module, ir.Module):
      if module.context is self.ic.context:
        return module
      # TODO: Fix upstream so that parse can accept bytes and then enable
      # binary=True.
      module = module.operation.get_asm(enable_debug_info=True)
    new_module = ir.Module.parse(module, context=self.ic.context)
    return new_module

  def import_global(self,
                    py_value: Any,
                    identifier: str,
                    *,
                    visibility: str = "private",
                    mutable: bool = True,
                    initialize: bool = False) -> ImportedGlobal:
    if identifier in self.globals:
      raise ValueError(f"Attempt to define duplicate global {identifier}")
    ic = self.ic
    assert not initialize, "TODO: Global initializer NYI"
    imp_value = ic.hooks.import_python_value(py_value)
    ir_types = ic.hooks.map_value(ic, imp_value)
    ops: List[iree_input_d.GlobalOp] = []
    symbols: List[ir.StringAttr] = []

    def emit(symbol: str, ir_type: ir.Type):
      with ic.loc, ic.ip:
        op = iree_input_d.GlobalOp(
            sym_visibility=ir.StringAttr.get(visibility),
            sym_name=ir.StringAttr.get(symbol),
            type=ir.TypeAttr.get(ir_type),
            is_mutable=ir.UnitAttr.get() if mutable else None,
            initializer=None,
            initial_value=None,
        )
        ops.append(op)
        ic.symbol_table.insert(op)
        # Must get the symbol name after insert, since it may be renamed.
        symbols.append(ir.StringAttr(op.attributes["sym_name"]))

    if len(ir_types) == 1:
      symbol = identifier
      emit(symbol, ir_types[0])
    else:
      for it, ir_type in enumerate(ir_types):
        symbol = f"{identifier}${it}"
        emit(symbol, ir_type)
    imported_global = ImportedGlobal(identifier, py_value, tuple(symbols),
                                     tuple(ops))
    self.globals[identifier] = imported_global
    return imported_global

  def import_globals(self,
                     linear_py_values: Sequence[Any],
                     identifier: str,
                     *,
                     mutable: bool = True,
                     initialize: bool = False) -> ImportedGlobalGroup:
    if identifier in self.global_groups:
      raise ValueError(f"Attempt to define duplicate global {identifier}")
    globals: List[ImportedGlobal] = list()
    for it, py_value in enumerate(linear_py_values):
      symbol = f"{identifier}${it}"
      globals.append(
          self.import_global(py_value,
                             symbol,
                             mutable=mutable,
                             initialize=initialize))
    global_group = ImportedGlobalGroup(identifier, tuple(linear_py_values),
                                       tuple(globals))
    self.global_groups[identifier] = global_group
    return global_group

  def get_function_type(self, name: str) -> ir.FunctionType:
    """Gets the FunctionType for a defined function by name."""
    func_op = self.ic.symbol_table[name]
    return func_op.type

  def create_function(self, name: str,
                      argument_types: Sequence[ir.Type]) -> "FunctionBuilder":
    ic = self.ic
    if name in ic.symbol_table:
      raise KeyError(f"Cannot create function '{name}': Already exists")
    with ic.loc, ic.ip:
      ftype = ir.FunctionType.get(argument_types, [])
      func_op = builtin_d.FuncOp(name, ftype)
      func_op.add_entry_block()
      ic.symbol_table.insert(func_op)
      return FunctionBuilder(self, func_op)


class FunctionBuilder:
  """Helpers for populating a function."""

  def __init__(self, mb: ModuleBuilder, func_op: builtin_d.FuncOp):
    self.mb = mb
    self.ic = self.mb.ic
    self.func_op = func_op
    self.ip = ir.InsertionPoint(self.func_op.entry_block)
    self.return_types = None

  @property
  def arguments(self) -> Sequence[ir.Value]:
    return list(self.func_op.entry_block.arguments)

  def emit_return(self, *values: Sequence[ir.Value]):
    ic = self.ic
    with ic.loc, self.ip:
      std_d.ReturnOp(values)
      # Check or rewrite the function return type.
      value_types = [v.type for v in values]
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

  def emit_call(self, name: str,
                arguments: Sequence[ir.Value]) -> Sequence[ir.Value]:
    ic = self.ic
    target_ftype = self.mb.get_function_type(name)
    with ic.loc, self.ip:
      return std_d.CallOp(target_ftype.results, name, arguments).results

  def emit_global_group_store(self, global_group: ImportedGlobalGroup,
                              flat_values: Sequence[ir.Value]):
    flattened_symbol_refs = global_group.flattened_symbol_refs
    if len(flattened_symbol_refs) != len(flat_values):
      raise ValueError(f"Mismatched arity in emit_global_group_store: "
                       f"global arity={len(flattened_symbol_refs)}, "
                       f"store arity={len(flat_values)}")
    ic = self.ic
    with ic.loc, self.ip:
      for symbol_ref, value in zip(flattened_symbol_refs, flat_values):
        iree_input_d.GlobalStoreOp(value=value, global_=symbol_ref)

  def emit_global_group_load(
      self, global_group: ImportedGlobalGroup) -> Sequence[ir.Value]:
    ic = self.ic
    flattened_symbol_refs = global_group.flattened_symbol_refs
    flattened_types = global_group.flattened_types
    results = []
    with ic.loc, self.ip:
      for symbol_ref, ir_type in zip(flattened_symbol_refs, flattened_types):
        results.append(iree_input_d.GlobalLoadOp(ir_type, symbol_ref).result)
    return results
