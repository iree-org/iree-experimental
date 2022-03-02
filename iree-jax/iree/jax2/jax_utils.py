# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Optional, Tuple, Union

from . import array_types

from iree.compiler import (
    ir,
    passmanager,
)
from iree.compiler.transforms import (
    ireec,)

import jax.core
import jax.interpreters.mlir
import jax.numpy as jnp

# Need to interop with the JAX version of MLIR, which may or may not be
# what we have here.
from jax._src.lib.mlir import ir as jax_ir
from jax.interpreters.xla import abstractify as jax_abstractify

_JAX_CONTEXT = jax_ir.Context()
_JAX_LOC = jax_ir.Location.unknown(context=_JAX_CONTEXT)


def aval_to_ir_types(context: ir.Context,
                     aval: jax.core.AbstractValue) -> Tuple[ir.Type]:
  # We use a Jax internal to do this, since it has the best knowledge.
  # However, this is very likely crossing a context/ABI boundary, so be
  # mindful and trip through text.
  # TODO: We could detect if these are actually the same instance and
  # elide this.
  with _JAX_LOC:
    jax_types = jax.interpreters.mlir.aval_to_ir_types(aval)

  def convert(jax_type: jax_ir.Type) -> ir.Type:
    return ir.Type.parse(str(jax_type), context=context)

  return tuple(convert(t) for t in jax_types)


def cleanup_mhlo_module(module: ir.Module):
  with module.context:
    pm = passmanager.PassManager()
    ireec.build_xla_cleanup_pass_pipeline(pm)
    # TODO: Don't lower it all the way here - but need to land bug fixes
    # first.
    #driver.build_mhlo_import_pass_pipeline(pm)
    pm.run(module)


def abstractify(x) -> jax.core.AbstractValue:
  # TODO: Ugh.
  if isinstance(x, jax.core.ConcreteArray):
    x = x.val
  if isinstance(x, array_types.TracedArrayBase):
    return x.aval
  # Note that a ConcreteArray is an AbstractValue so we handle that above.
  if isinstance(x, jax.core.AbstractValue):
    return x
  return jax_abstractify(x)


def unwrap_global_array(x) -> Optional[array_types.ExportedGlobalArray]:
  # TODO: Ugh. Ugh.
  if isinstance(x, jax.core.ConcreteArray):
    x = x.val
  if not isinstance(x, array_types.ExportedGlobalArray):
    return None
  return x


def import_module(context: ir.Context, module: Union[str, ir.Module]):
  if isinstance(module, ir.Module):
    if module.context is context:
      return module
    # TODO: Fix upstream so that parse can accept bytes and then enable
    # binary=True.
    module = module.operation.get_asm(enable_debug_info=True)

  if not isinstance(module, str):
    raise ValueError(
        f"Attempted to import a non-module (did you enable MLIR in JAX?). "
        f"Got {module}")
  new_module = ir.Module.parse(module, context=context)
  return new_module


def import_main_function(*,
                         target_module: ir.Module,
                         target_symbol_table: ir.SymbolTable,
                         source_module: Union[str, ir.Module],
                         main_symbol: str = "main",
                         visibility: str = "private") -> str:
  """Imports a named function from another module into this one.

  Returns (imported symbol name, operation) of the found function (if
  present).

  TODO: This is horrible. Burn it.
  """
  context = target_module.context
  source_module = import_module(context, source_module)
  cleanup_mhlo_module(source_module)

  with context:
    target_body = target_module.body
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
