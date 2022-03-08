# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

from . import array_types
from . import ir_utils
from . import jax_utils
from . import tracing

from iree.compiler.dialects import (
    func as func_d,)

import jax
import jax.core
from jax.tree_util import (tree_map, tree_flatten, tree_unflatten)

__all__ = [
    "export_pure_func",
    "store_global",
]

logger = logging.getLogger("iree_jax")


def _instantiate(cls):
  return cls()


class export_pure_func(tracing.CallableIntrinsic):
  """Decorator which makes a pure function callable from tracing.

  Internally, it will be JIT-ed and exported as needed.

  TODO: This should really be an internal detail of the exporter, not part of
  the user API. It is just here to facilite an initial prototype.
  """

  def __init__(self, wrapped_f, *, wrap_with_jit: bool = True):
    self.wrapped_f = wrapped_f
    self.jit_f = jax.jit(self.wrapped_f) if wrap_with_jit else self.wrapped_f

  def __repr__(self):
    return f"<Exportable Pure Func: {self.wrapped_f}>"

  def resolve_call(self, func_trace: tracing.FunctionIrTrace, *args):
    abstract_args = tree_map(lambda x, *xs: jax_utils.abstractify(x), args)
    # TODO: Requires local patch to JAX. Should fix this to be generic.
    # Requires JAX_ENABLE_MLIR=1 in the environment.
    # TODO: We really should be suspending the tracing context here so that
    # recursive calls to functions that may be traced happen as if outside.
    lowered = self.jit_f.lower(*abstract_args)
    result_tree_def = lowered.out_tree
    lowered_asm = str(lowered.compiler_ir())
    imported_main_symbol_name = jax_utils.import_main_function(
        target_module=func_trace.module,
        target_symbol_table=func_trace.module_symbol_table,
        source_module=lowered_asm)

    # TODO: Signficiant verification could be done here in order to save
    # trouble down stream and emit errors at the point they are made.
    target_ftype = ir_utils.get_function_type(func_trace.module_symbol_table,
                                              imported_main_symbol_name)
    logging.debug("Emitting call to kernel: %s", target_ftype)

    # TODO: Another magic stashed argument (position 6). Make a real name for
    # this.
    kept_var_idx = lowered._lowering.compile_args["kept_var_idx"]
    assert len(kept_var_idx) == len(target_ftype.inputs), (
        f"Mismatched arguments in Jax kept_var_idx vs func decl: "
        f"{len(kept_var_idx)} vs {len(target_ftype.inputs)}")

    # Flatten and convert args to IR values.
    flat_py_args, _ = tree_flatten(args)
    flat_ir_args = []
    for idx, py_arg in enumerate(flat_py_args):
      if idx not in kept_var_idx:
        logger.debug("Skipping pruned argument %d=(of type %r)", idx,
                    type(py_arg))
        continue
      logger.debug("Materializing call arg to IR: %r (%r)", py_arg,
                  type(py_arg))
      flat_ir_args.extend(func_trace.materialize_py_values(py_arg))

    assert len(flat_ir_args) == len(target_ftype.inputs), (
        f"Mismatched number of IR call args vs function decl: "
        f"{len(flat_ir_args)} vs {len(target_ftype.inputs)}\n"
        f"  For call to: {target_ftype}\n"
        f"  From: {flat_py_args}\n")

    flat_results_ir = func_d.CallOp(target_ftype.results,
                                   imported_main_symbol_name,
                                   flat_ir_args).results

    # Now convert each IR result to an intrinsic.
    # TODO: Switch based on values not an array?
    # TODO: Better way to get the lowering abstract values. See:
    #   https://github.com/google/jax/issues/8745
    flat_results_aval = lowered._lowering.compile_args["out_avals"]
    flat_results_py = map(
        lambda aval, ir_value: array_types.IrValueArray(aval, ir_value),
        flat_results_aval, flat_results_ir)

    # Put it back in a tree.
    results_tree = tree_unflatten(result_tree_def, flat_results_py)
    return results_tree


@_instantiate
class store_global(tracing.CallableIntrinsic):
  """Stores a source tree into a target tree.

  The target tree must have previously been exported.
  """

  def resolve_call(self, func_trace: tracing.FunctionIrTrace, target_tree,
                   source_tree):
    target_flat, target_tree_def = tree_flatten(target_tree)
    source_flat, source_tree_def = tree_flatten(source_tree)
    if target_tree_def != source_tree_def:
      raise TypeError(f"Attempt to store mismatched tree\n"
                      f"  {source_tree}\n"
                      f"    (described by {source_tree_def})\n"
                      f"into\n"
                      f"  {target_tree}\n"
                      f"    (described by {target_tree_def})")

    # Resolve symbols.
    target_global_symbols = []
    for target_item in target_flat:
      global_array = jax_utils.unwrap_global_array(target_item)
      if not global_array:
        raise TypeError(f"'store_global' expects a tree of global arrays. "
                        f"Got:\n  {target_flat}")
      target_global_symbols.append(global_array.symbol_name)

    # Resolve each and store.
    # TODO: Verify some more type invariants as the error messages we can
    # emit here will be better than if it fails verification down the line.
    for symbol_name, source_intrinsic in zip(target_global_symbols,
                                             source_flat):
      source_values = func_trace.materialize_py_values(source_intrinsic)
      assert len(source_values) == 1
      ir_utils.create_global_store_op(symbol_name, source_values[0])
