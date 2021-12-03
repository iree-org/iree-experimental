# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

logging.basicConfig(level=logging.DEBUG)

from iree.jax2.staging_api import *
from iree.jax2.builtins import *

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_mlir", True)

a = jnp.zeros((3, 4), jnp.float32)
b = jnp.zeros((3, 4), jnp.float32)

params = {"a": a, "b": b}


class LineModule(StagedModule):

  params = export_global(params)

  @export_kernel
  def linear(_, m, x, b):
    return m * x + b

  @export_traced_proc(signature=[a])
  def run(mdl, multiplier):
    result = mdl.linear(multiplier, mdl.params["a"], mdl.params["b"])
    store_global(mdl.params["a"], result)

  @export_traced_proc
  def get_params(mdl):
    return mdl.params


print(get_mlir_module(LineModule))

lm = LineModule()
