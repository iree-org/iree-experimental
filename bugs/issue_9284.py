# RUN: %PYTHON %s
# XFAIL: *

import iree.compiler
import iree.jax
import jax
import jax.numpy as jnp

# https://github.com/iree-org/iree/issues/9284

x = jnp.ones((2, 5), dtype=jnp.float32)

class BugTest(iree.jax.Program):
  @iree.jax.kernel
  def _test(x):
    return jax.lax.conv_general_dilated_patches(
                lhs=x,
                filter_shape=(),
                window_strides=[],
                padding=[],
                dimension_numbers=("NC", "OI", "CN"),
                precision=jax.lax.Precision.DEFAULT
            )

  def test(self, x=iree.jax.like(x)):
    return self._test(x)

ir = str(iree.jax.Program.get_mlir_module(BugTest()))
print(ir)

# Compilation fails.
iree.compiler.compile_str(ir, target_backends=["cpu"], input_type="mhlo")