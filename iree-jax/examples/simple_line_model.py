# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import namedtuple
import logging

from iree.jax2.staging_api import *
from iree.jax2.builtins import *

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_mlir", True)

a = jnp.zeros((3, 4), jnp.float32)
b = jnp.zeros((3, 4), jnp.float32)

Params = namedtuple("Params", "a,b")

params = Params(a, b)


class LineModule(StagedModule):

  # TODO: Add trivial initialization support.
  _params = export_global(params)

  @export_kernel
  def linear(_, m, x, b):
    return m * x + b

  @export_traced_proc(signature=[a])
  def run(mdl, multiplier):
    result = mdl.linear(multiplier, mdl._params.a, mdl._params.b)
    store_global(mdl._params.a, result)

  @export_traced_proc
  def get_params(mdl):
    return mdl._params


print(get_mlir_module(LineModule))

lm = LineModule()
