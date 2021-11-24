# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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


class ModuleBuilder:
  """Helpers for constructing an IREE input module."""

  def __init__(self, ic: Optional[ir_utils.ImportContext] = None):
    self.ic = ic if ic else ir_utils.ImportContext()
    self.ic.push_ip(ir.InsertionPoint(self.ic.module_body))

  def __str__(self):
    return str(self.ic)

  def import_function(
      self,
      module: Union[str, ir.Module],
      main_symbol: str,
      visibility: str = "private"
  ) -> Tuple[Optional[str], Optional[ir.Operation]]:
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
    return found_name, found_function

  def load_module(self, module: Union[str, ir.Module]):
    if isinstance(module, ir.Module):
      if module.context is self.ic.context:
        return module
      module = str(module)
    new_module = ir.Module.parse(module, context=self.ic.context)
    return new_module

  def import_global(self,
                    py_value: Any,
                    symbol_prefix: str,
                    *,
                    visibility: str = "private",
                    mutable: bool = True,
                    initialize: bool = False) -> Tuple[iree_input_d.GlobalOp]:
    ic = self.ic
    assert not initialize, "TODO: Global initializer NYI"
    imp_value = ic.hooks.import_python_value(py_value)
    ir_types = ic.hooks.map_value(ic, imp_value)
    ops: List[iree_input_d.GlobalOp] = []

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

    if len(ir_types) == 1:
      symbol = symbol_prefix
      emit(symbol, ir_types[0])
    else:
      for it, ir_type in enumerate(ir_types):
        symbol = f"{symbol_prefix}${it}"
        emit(symbol, ir_type)
    return tuple(ops)

  def import_globals(
      self,
      linear_py_values: Sequence[Any],
      symbol_prefix: str,
      *,
      mutable: bool = True,
      initialize: bool = False) -> Sequence[Tuple[iree_input_d.GlobalOp]]:
    ops = []
    for it, py_value in enumerate(linear_py_values):
      symbol = f"{symbol_prefix}${it}"
      ops.append(
          self.import_global(py_value,
                             symbol,
                             mutable=mutable,
                             initialize=initialize))
    return ops

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

  def emit_call(self, name: str,
                arguments: Sequence[ir.Value]) -> Sequence[ir.Value]:
    ic = self.ic
    target_ftype = self.mb.get_function_type(name)
    with ic.loc, self.ip:
      return std_d.CallOp(target_ftype.results, name, arguments).results

  def emit_group_global_store(self, global_group, flat_values: ir.Value):
    ic = self.ic
    flat_value_len = len(flat_values)
    flat_value_index = 0
    with ic.loc, self.ip:
      for global_comp in global_group:
        for global_op in global_comp:
          if flat_value_index >= flat_value_len:
            raise IndexError(
                f"Mismatched group_global_store arity {flat_value_index}")
          symbol = ir.StringAttr(global_op.attributes["sym_name"]).value
          symbol_ref = ir.FlatSymbolRefAttr.get(symbol)
          iree_input_d.GlobalStoreOp(value=flat_values[flat_value_index],
                                     global_=symbol_ref)
          flat_value_index += 1
    if flat_value_index != flat_value_len:
      raise IndexError(f"Mismatched group_global_store arity")
