# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler import ir
from iree.nn.jax_utils import (
    JaxImportContext,)
from iree.nn.module_builder import (
    ModuleBuilder,)

import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax

from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           register_pytree_node)


def run(f):
  print(f"BEGIN: {f.__name__}")
  context = ir.Context()
  with context:
    f()
  print(f"END: {f.__name__}\n")
  return f


@run
def basic():
  imp = ModuleBuilder(JaxImportContext())
  print(imp)


@run
def jax_params():
  imp = ModuleBuilder(JaxImportContext())
  init_random_params, predict = stax.serial(Dense(1024), Relu, Dense(1024),
                                            Relu, Dense(10), LogSoftmax)
  rng = random.PRNGKey(0)
  _, init_params = init_random_params(rng, (-1, 28 * 28))
  opt_init, opt_update, get_params = optimizers.momentum(0.001, mass=0.9)
  opt_state = opt_init(init_params)

  print(init_params.__class__)
  print(opt_state.__class__)

  opt_state_flat, tree_def = tree_flatten(opt_state)
  opt_state_globals = imp.import_globals(opt_state_flat,
                                         "opt_state",
                                         mutable=True,
                                         initialize=False)
  print(opt_state_globals)

  @jit
  def initialize(rng):
    _, init_params = init_random_params(rng, (-1, 28 * 28))
    return opt_init(init_params)
    #return init_params

  mhlo_text = initialize.lower(rng)._xla_computation()
  f_name, f = imp.import_function(mhlo_text, "main")

  fb = imp.create_function("initialize", imp.get_function_type(f_name).inputs)
  initial_values = fb.emit_call(f_name, fb.arguments)
  fb.emit_group_global_store(opt_state_globals, initial_values)
  fb.emit_return()

  # print(opt_state_flat)
  # unflatten = tree_unflatten(tree_def, opt_state_flat)
  # print(unflatten)

  print(imp)
