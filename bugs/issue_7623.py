# RUN: %PYTHON %s
# XFAIL: *

import iree.compiler
import iree.jax
import jax.lax
import jax.numpy as jnp

# https://github.com/iree-org/iree/issues/9669

arg0 = jnp.ones((16,), dtype=jnp.float32)
arg1 = jnp.ones((16,), dtype=jnp.float32)

# Attempting to replicate fails below.
class BugTest(iree.jax.Program):
  @iree.jax.kernel
  def _test(a0, a1):
    return jax.lax.complex(a0, a1)

  def test(self, a0=iree.jax.like(arg0), a1=iree.jax.like(arg1)):
    return self._test(a0, a1)

ir = str(iree.jax.Program.get_mlir_module(BugTest()))
print(ir)

# Compilation fails.
iree.compiler.compile_str(ir, target_backends=["cpu"], input_type="mhlo")