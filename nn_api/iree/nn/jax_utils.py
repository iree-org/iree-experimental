# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Optional, Tuple
from .ir_utils import ImportContext, ImportHooks, ImportedValue

from iree.compiler import (
    ir,
    passmanager,
)
from iree.compiler.api import (
    driver,)

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

import jax.core
import jax.interpreters.mlir

# Need to interop with the JAX version of MLIR, which may or may not be
# what we have here.
from jax._src.lib.mlir import ir as jax_ir

# TODO: Should be a common utility?
from jax.interpreters.xla import abstractify


class JaxImportHooks(ImportHooks):
  """Import hooks that register JAX type mappings."""

  def __init__(self):
    super().__init__()
    self._jax_context = jax_ir.Context()
    self._jax_loc = jax_ir.Location.unknown(context=self._jax_context)
    self.add_type_mapper(jax.core.AbstractValue, self._aval_to_ir_types)

  def preprocess_module(self, ic: "ImportContext", module: ir.Module):
    """Performs preprocessing cleanup tasks on a module."""
    _cleanup_mhlo_module(module)

  def import_python_value(self, py_value: Any) -> ImportedValue:
    try:
      aval = abstractify(py_value)
      return ImportedValue(aval, py_value)
    except TypeError:
      return super().import_python_value(py_value)

  def _aval_to_ir_types(self, ic: ImportContext,
                        imp_val: ImportedValue) -> Tuple[ir.Type]:
    # We use a Jax internal to do this, since it has the best knowledge.
    # However, this is very likely crossing a context/ABI boundary, so be
    # mindful and trip through text.
    # TODO: We could detect if these are actually the same instance and
    # elide this.
    with self._jax_loc:
      jax_types = jax.interpreters.mlir.aval_to_ir_types(imp_val.imported)

    def convert(jax_type: jax_ir.Type) -> ir.Type:
      return ir.Type.parse(str(jax_type), context=ic.context)

    return tuple(convert(t) for t in jax_types)


class JaxImportContext(ImportContext):
  """Jax specific import context."""

  def create_hooks(self) -> ImportHooks:
    return JaxImportHooks()

  def create_context(self, **kwargs) -> ir.Context:
    context = super().create_context(**kwargs)
    chlo_d.register_chlo_dialect(context)
    mhlo_d.register_mhlo_dialect(context)
    return context


def _import_type(context: ir.Context, foreign_type: Any) -> ir.Type:
  if foreign_type.context is context:
    return foreign_type
  return ir.Type.parse(str(foreign_type), context=context)


def _cleanup_mhlo_module(module: ir.Module):
  with module.context:
    pm = passmanager.PassManager()
    driver.build_xla_cleanup_pass_pipeline(pm)
    driver.build_mhlo_import_pass_pipeline(pm)
    pm.run(module)
