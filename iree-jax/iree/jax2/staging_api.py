# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""High level API for constructing input program modules.

This is intended to be the primary entry point for staging out a Jax program.
It interfaces with the lower level exporter.
"""

import contextlib
import functools
import logging
import re
from typing import Any, ClassVar, Optional, Sequence, Type, Union

from .builtins import (
    export_pure_func,)
from .exporter import ExportModule

from iree.compiler import (
    ir,
    tools as iree_tools,
)

import jax

__all__ = [
    "export_global",
    "export_kernel",
    "export_traced_proc",
    "get_compiled_binary",
    "get_mlir_module",
    "StagedModule",
]

logger = logging.getLogger("iree_jax")

################################################################################
# StagedModule (and metaclass)
################################################################################

_allow_user_subclasses = False


class StagedModuleMeta(type):
  """Meta class for all modules.

  Do not use directly (subclass StagedModule).
  """

  def __new__(mcls,
              name: str,
              bases,
              dct,
              *,
              context: Optional[ir.Context] = None,
              export_name: Optional[str] = None):
    if not _allow_user_subclasses:
      # Still defining this API, so not creating user subclasses yet.
      return type.__new__(mcls, name, bases, dct)

    export_name = _derive_module_export_name(name, export_name)
    export_module = ExportModule.create_empty(context=context, name=export_name)

    logger.debug("Create new StagedModule: %s", export_name)
    updates = {}
    class_callbacks = []

    for key, attr in dct.items():
      if key.startswith("__") and key.endswith("__"):
        continue
      if isinstance(attr, PrivateKernelDescriptor):
        mcls._bind_private_kernel(export_module, key, attr, class_callbacks)
      elif isinstance(attr, ExportedGlobalDescriptor):
        mcls._bind_exported_global(export_module, key, attr)
      elif isinstance(attr, TracedProcDescriptor):
        mcls._bind_traced_proc(export_module, key, attr, class_callbacks)
      else:
        raise AttributeError(f"Unsupported attribte on StagedModule subclass: "
                             f"{key} = {attr}")

    dct["_export_module"] = export_module
    dct.update(updates)
    new_class = type.__new__(mcls, name, bases, dct)

    for callback in class_callbacks:
      callback(new_class)

    return new_class

  def _bind_exported_global(export_module: ExportModule, attr_name: str,
                            d: "ExportedGlobalDescriptor"):
    assert d.tracked_value is None, (
        "'export_global' stored on class multiple times")
    logger.debug("Found exported global: %r", attr_name)
    export_name = d.export_name if d.export_name is not None else attr_name
    d.export_name = export_name
    # TODO: Need to differentiate between single array and tree.
    tracked = export_module.def_global_tree(export_name,
                                            d.captured_value,
                                            initialize=d.initialize,
                                            mutable=d.mutable)
    d.tracked_value = tracked

  def _bind_private_kernel(export_module: ExportModule, attr_name: str,
                           d: "PrivateKernelDescriptor", class_callbacks):
    assert d.tracked_value is None, (
        "'export_kernel' stored on class multiple times")
    d.name = attr_name

    def post_process_with_cls(cls):

      def invoke_with_cls(*args, **kwargs):
        return d.raw_f(cls, *args, **kwargs)

      jitted_f = jax.jit(invoke_with_cls)
      d.tracked_value = export_pure_func(jitted_f, wrap_with_jit=False)

    class_callbacks.append(post_process_with_cls)

  def _bind_traced_proc(export_module: ExportModule, attr_name: str,
                        d: "TracedProcDescriptor", class_callbacks):
    assert d.tracked_value is None, (
        "'export_traced_proc' stored on class multiple times")
    logger.debug("Found traced proc: %r", attr_name)
    export_name = d.export_name if d.export_name is not None else attr_name
    d.export_name = export_name

    def post_process_with_cls(cls):

      def invoke_with_cls(*args, **kwargs):
        return d.raw_f(cls, *args, **kwargs)

      d.tracked_value = export_module.def_func(invoke_with_cls,
                                               symbol_name=d.export_name,
                                               arguments=d.signature)

    class_callbacks.append(post_process_with_cls)


class StagedModule(metaclass=StagedModuleMeta):
  """Base class for all user-defined staged modules."""
  _export_module: ClassVar[ExportModule]

  def __init__(self):
    self._compiled_module = CompiledModule(self)


# Now enable any new subclasses with a StagedModuleMeta metaclass to be treated
# as user classes.
_allow_user_subclasses = True

StagedModuleClassOrInstance = Union[StagedModule, Type[StagedModule]]

################################################################################
# Compiled Module
################################################################################


class CompiledModule:
  """A StagedModule that has undergone compilation."""

  def __init__(self, staged_module: StagedModule):
    # TODO: Need a lot more options here.
    # TODO: We should be storing these in a cache, backing them by files that
    # are mmap'd in.
    logger.debug("Compiling module...")
    self._binary = iree_tools.compile_str(str(get_mlir_module(staged_module)),
                                          target_backends=["cpu"],
                                          input_type="mhlo")


################################################################################
# Accessors for the StagedModule
# We have these as free-standing functions so as to avoid namespace pollution.
################################################################################


def get_mlir_module(staged_module: StagedModuleClassOrInstance) -> ir.Module:
  return staged_module._export_module.module


def get_compiled_binary(staged_module: StagedModule):
  return staged_module._compiled_module._binary


################################################################################
# Decorators and helpers
################################################################################


def export_global(captured_value: Any,
                  *,
                  export_name: Optional[str] = None,
                  initialize: bool = False,
                  mutable: bool = True):
  # TODO: Should differentiate between single value and tree.
  return ExportedGlobalDescriptor(captured_value,
                                  export_name=export_name,
                                  initialize=initialize,
                                  mutable=mutable)


def export_traced_proc(f=None,
                       *,
                       signature: Sequence = (),
                       export_name: Optional[str] = None):
  if f is None:
    return functools.partial(export_traced_proc,
                             signature=signature,
                             export_name=export_name)

  return TracedProcDescriptor(f, signature=signature, export_name=export_name)


def export_kernel(f=None):
  if f is None:
    return functools.partial(f)
  return PrivateKernelDescriptor(f)


################################################################################
# Descriptors
################################################################################


class BaseDescriptor:
  """Base class for data descriptors that we own."""
  __slots__ = []


class ExportedGlobalDescriptor(BaseDescriptor):
  """A descriptor for an exported global."""
  __slots__ = [
      "captured_value",
      "initialize",
      "mutable",
      "export_name",
      "tracked_value",
  ]

  def __init__(self, captured_value: Any, *, export_name: Optional[str],
               initialize: bool, mutable: bool):
    self.captured_value = captured_value
    self.export_name = export_name
    self.initialize = initialize
    self.mutable = mutable
    self.tracked_value = None

  def __get__(self, obj, objtype=None):
    if obj is not None:
      raise ValueError(
          f"The global '{self.export_name}' was exported privately "
          "and cannot be resolved on a compiled module instance")
    return self.tracked_value


class TracedProcDescriptor(BaseDescriptor):
  """Descriptor wrapping a traced procedure."""
  __slots__ = [
      "export_name",
      "raw_f",
      "signature",
      "tracked_value",
  ]

  def __init__(self, f, *, signature: Sequence, export_name: Optional[str]):
    self.raw_f = f
    self.signature = signature
    self.export_name = export_name
    self.tracked_value = None

  def __get__(self, obj, objtype=None):
    if obj is None:
      return self.tracked_value
    else:
      raise ValueError(f"Calls to a traced proc not yet implemented")


class PrivateKernelDescriptor(BaseDescriptor):
  """Descriptor that declares a Jax jittable kernel function exported.

  This will be auto-exported on use. The kernel will implicitly have the
  class passed as its first argument.
  """
  __slots__ = [
      "name",
      "raw_f",
      "tracked_value",
  ]

  def __init__(self, raw_f):
    self.raw_f = raw_f
    self.name = None
    self.tracked_value = None

  def __get__(self, obj, objtype=None):
    if obj is not None:
      raise ValueError(f"The Jax function '{self.name}' was exported privately "
                       "and cannot be resolved on a compiled module instance")
    return self.tracked_value


################################################################################
# Private utilities
################################################################################


def _derive_module_export_name(class_name: str, explicit_name: Optional[str]):
  """Returns an appropriate module export name given a class name and override.

  If an explicit_name is given, that is used as is. Otherwise, the class name
  is mangled by:
    * Removing and "Module" suffix.
    * Converting camel case to snake case.
  """
  if explicit_name:
    return explicit_name
  return _to_snake_case(_strip_suffix(class_name, "Module"))


def _to_snake_case(s: str) -> str:
  return re.sub(r'(?<!^)(?=[A-Z])', '_', s).lower()


def _strip_suffix(s: str, optional_suffix: str) -> str:
  if s.endswith(optional_suffix):
    return s[0:len(s) - len(optional_suffix)]
  else:
    return s
