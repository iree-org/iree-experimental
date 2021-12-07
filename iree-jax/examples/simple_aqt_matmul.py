# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Test compiling and executing a basic AQT MatMul with IREE."""

import unittest

import jax
import jax.numpy as jnp

BACKEND = "llvmaot"


def aqt_matmul(activation, weight, activation_scale):
  precision = 8
  lower_bound = -2**(precision - 1) + 1
  upper_bound = 2**(precision - 1) - 1

  activation_scaled = activation * activation_scale
  activation_rounded = jnp.floor(activation_scaled + jnp.array(0.5))
  activation_clipped = jnp.clip(activation_rounded, lower_bound, upper_bound)
  activation_as_int = activation_clipped.astype(jnp.int8)

  weight_scale = upper_bound / jnp.max(jnp.abs(weight))
  weight_scaled = weight * weight_scale
  weight_rounded = jnp.floor(weight_scaled + jnp.array(0.5))
  weight_as_int = weight_rounded.astype(jnp.int8)

  scaled_result = jax.lax.dot(
      activation_as_int, weight_as_int, preferred_element_type=jnp.int32)
  return scaled_result / (activation_scale * weight_scale)


class AQTMatmulTest(unittest.TestCase):

  def test_aqt_matmul(self):
    activation = jnp.arange(30, dtype=jnp.float32).reshape(5, 6) / 10.4
    activation_scale = jnp.array(5.0)
    weight = jnp.arange(18, dtype=jnp.float32).reshape(6, 3) * 500.3
    utils.compile_and_assert_all_close(
        BACKEND,
        aqt_matmul,
        activation,
        weight,
        activation_scale,
        artifact_name="aqt_matmul")


if __name__ == "__main__":
  unittest.main()
