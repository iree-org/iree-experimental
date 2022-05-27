# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections import namedtuple
import logging

from iree.jax import Program, kernel, like

import jax
import jax.numpy as jnp

logging.basicConfig(level=logging.DEBUG)

a = jnp.zeros((3, 4), jnp.float32)
b = jnp.zeros((3, 4), jnp.float32)

Params = namedtuple("Params", "a,b")
params = Params(a, b)

class LineModule(Program):

  _params = params

  @kernel
  def linear(m, x, b):
    return m * x + b

  def run(self, multiplier=like(a)):
    result = self.linear(multiplier, self._params.a, self._params.b)
    print(self._params.a)
    self._params = Params(result, self._params.b)

  def get_params(self):
    return self._params


print(Program.get_mlir_module(LineModule))

lm = LineModule()
