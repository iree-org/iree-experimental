# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import jax
import numpy as np

from ..util import simple_backend

simple_backend.register_backend()

from absl import logging

logging.set_verbosity(10)


@jax.jit
def f(x, y):
  return x + y - 3


print(f(2, 77))
print(f(np.asarray([2, 3]), np.asarray([77])))
print(f(np.asarray([5, 4]), np.asarray([77])))
print(f(np.asarray([6, 5]), np.asarray([77])))
print(f(np.asarray([7, 6]), np.asarray([77])))
