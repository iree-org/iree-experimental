# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Base helpers for the Simple Lang DSL."""

from typing import List, Optional

import io
import functools
import sys

from iree.compiler.api import driver
from iree.compiler.dialects.iree_pydm.importer import (
    create_context,
    def_pyfunc_intrinsic,
    DefaultImportHooks,
    FuncProvidingIntrinsic,
    ImportContext,
    ImportStage,
)
from iree.compiler.dialects.iree_pydm.rtl import (
    get_std_rtl_asm,
)

from iree.compiler.dialects import builtin as builtin_d
from iree.compiler.dialects import iree_pydm as pydm_d
from iree.compiler import (ir, passmanager, transforms as unused_transforms)
from iree import runtime as iree_runtime
from iree.runtime.system_api import load_vm_module


class SimpleModule:
  """Declares a runtime library module.

  This is typically included in a Python module which exports functions as:

  ```
  M = SimpleModule("foo")

  @M.export_pyfunc
  def object_as_bool(v) -> bool:
    ...
  ```
  """

  def __init__(self, name: str = "module"):
    self.name = name
    self.exported_funcs: List[FuncProvidingIntrinsic] = []
    self._compiled_binary = None
    self._loaded_module = None

  def export_pyfunc(self,
                    f=None,
                    *,
                    symbol: Optional[str] = None,
                    visibility: Optional[str] = None):
    """Marks a python function for export in the created module.

    This is typically used as a decorator and returns a FuncProvidingIntrinsic.
    """
    if f is None:
      return functools.partial(
          self.export_pyfunc, symbol=symbol, visibility=visibility)
    if symbol is None:
      symbol = f.__name__
    intrinsic = def_pyfunc_intrinsic(f, symbol=symbol, visibility=visibility)
    self.exported_funcs.append(intrinsic)
    return intrinsic

  def internal_pyfunc(self,
                      f=None,
                      *,
                      symbol: Optional[str] = None,
                      visibility: Optional[str] = "private"):
    if f is None:
      return functools.partial(
          self.internal_pyfunc, symbol=symbol, visibility=visibility)
    if symbol is None:
      symbol = f.__name__
    intrinsic = def_pyfunc_intrinsic(f, symbol=symbol, visibility=visibility)
    return intrinsic

  def compile(self, context: Optional[ir.Context] = None) -> "Compiler":
    """Compiles the module given the exported and internal functions."""
    compiler = Compiler(context)
    compiler.import_module(self)
    compiler.compile()
    self._compiled_binary = compiler.translate()
    return compiler

  @property
  def compiled_binary(self):
    if self._compiled_binary is None:
      self.compile()
    return self._compiled_binary

  @property
  def loaded_module(self):
    if self._loaded_module is None:
      # TODO: This API in IREE needs substantial ergonomic work for loading
      # a module from a memory image.
      system_config = _get_global_config()
      vm_module = iree_runtime.binding.VmModule.from_flatbuffer(
          self.compiled_binary)
      self._loaded_module = load_vm_module(vm_module, system_config)
    return self._loaded_module

  def save(self, filename: str):
    with open(filename, "wb") as f:
      f.write(self.compiled_binary)


class Compiler:
  """A module being compiled."""

  def __init__(self, context: Optional[ir.Context] = None, debug: bool = True):
    self.debug = debug
    self.context = context if context else create_context(debug=debug)
    self.hooks = DefaultImportHooks()
    self.root_module = ir.Module.create(
        ir.Location.unknown(context=self.context))
    self.module_op = self.root_module.operation
    self.rtl_asm = get_std_rtl_asm()

    # IREE compiler options.
    self.options = driver.CompilerOptions()
    self.options.add_target_backend("cpu")

  def __str__(self):
    return str(self.root_module)

  def import_module(self, m: SimpleModule):
    with self.context:
      root_body = self.module_op.regions[0].blocks[0]
      self.module_op.attributes["sym_name"] = ir.StringAttr.get(m.name)
    ic = ImportContext(context=self.context, module=self.module_op)
    stage = ImportStage(ic=ic, hooks=self.hooks)

    # Export functions.
    for f in m.exported_funcs:
      # Getting the symbol implies exporting it into the module.
      f.get_or_create_provided_func_symbol(stage)

  def compile(self):
    """Compiles the module."""
    with self.context:
      # TODO: Create a real pass pipeline to do first stage optimizations.
      pm = passmanager.PassManager.parse("builtin.module(canonicalize,cse)")
      if self.debug:
        pm.enable_ir_printing()
      pydm_d.build_lower_to_iree_pass_pipeline(pm, link_rtl_asm=self.rtl_asm)
      pm.run(self.root_module)
      #self.root_module.operation.print(enable_debug_info=True)

      pm = passmanager.PassManager()
      driver.build_iree_vm_pass_pipeline(self.options, pm)
      pm.run(self.root_module)

  def translate(self):
    """Translates to a binary, returning a buffer."""
    bytecode_io = io.BytesIO()
    driver.translate_module_to_vm_bytecode(self.options, self.root_module,
                                           bytecode_io)
    return bytecode_io.getbuffer()


# TODO: This is hoaky and needs to go in a real runtime layer.
_cached_global_config: Optional[iree_runtime.system_api.Config] = None


def _get_global_config() -> iree_runtime.system_api.Config:
  global _cached_global_config
  if not _cached_global_config:
    _cached_global_config = iree_runtime.system_api.Config("dylib")
  return _cached_global_config
