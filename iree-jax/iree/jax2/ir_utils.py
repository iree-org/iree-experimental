# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Sequence, Tuple, Union
from iree.compiler import (
    ir,)

from iree.compiler.dialects import (
    builtin as builtin_d,
    chlo as chlo_d,
    iree_input as iree_input_d,
    mhlo as mhlo_d,
)


def create_context(*, debug: bool = True) -> ir.Context:
  context = ir.Context()
  if debug:
    context.enable_multithreading(False)
  iree_input_d.register_dialect(context)
  chlo_d.register_chlo_dialect(context)
  mhlo_d.register_mhlo_dialect(context)
  return context


def create_global(symbol_table: ir.SymbolTable,
                  symbol: str,
                  ir_type: ir.Type,
                  *,
                  mutable: bool = True,
                  visibility: str = "private",
                  initial_value: Optional[ir.Attribute] = None) -> str:
  op = iree_input_d.GlobalOp(
      sym_visibility=ir.StringAttr.get(visibility),
      sym_name=ir.StringAttr.get(symbol),
      type=ir.TypeAttr.get(ir_type),
      is_mutable=ir.UnitAttr.get() if mutable else None,
      initializer=None,
      initial_value=initial_value,
  )
  symbol_table.insert(op)
  # Must get the symbol name after insert, since it may be renamed.
  # TODO: Wish there was a better API for this dance.
  return ir.StringAttr(op.attributes["sym_name"]).value


def create_func_op(
    symbol_table: ir.SymbolTable, symbol_name: str,
    argument_types: Sequence[ir.Type]) -> Tuple[str, builtin_d.FuncOp]:
  ftype = ir.FunctionType.get(argument_types, [])
  func_op = builtin_d.FuncOp(symbol_name, ftype)
  func_op.add_entry_block()
  symbol_table.insert(func_op)
  actual_symbol_name = ir.StringAttr(func_op.attributes["sym_name"]).value
  return actual_symbol_name, func_op


def create_global_load_op(symbol_name: str, ir_type: ir.Type) -> ir.Value:
  symbol_ref = ir.FlatSymbolRefAttr.get(symbol_name)
  return iree_input_d.GlobalLoadOp(ir_type, symbol_ref).result


def create_global_store_op(symbol_name: str, ir_value: ir.Value):
  symbol_ref = ir.FlatSymbolRefAttr.get(symbol_name)
  iree_input_d.GlobalStoreOp(value=ir_value, global_=symbol_ref)


def get_function_type(symbol_table: ir.SymbolTable, symbol_name: str) -> ir.FunctionType:
  func_op = symbol_table[symbol_name]
  # TODO: Verify that it is a function, etc.
  return ir.FunctionType(func_op.type)


def create_array_attribute(array, ir_types: Sequence[ir.Type]) -> ir.Attribute:
  if len(ir_types) != 1:
    raise ValueError("Only single-typed arrays are supported")
  ranked_tensor_type = ir.RankedTensorType(ir_types[0])
  return ir.DenseElementsAttr.get(array, type=ranked_tensor_type.element_type)
