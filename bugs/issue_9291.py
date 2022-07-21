# RUN: %PYTHON %s
# XFAIL: *

import iree.compiler
import iree.jax
import iree.runtime
import jax.lax
import jax.numpy as jnp
import numpy as np

# https://github.com/iree-org/iree/issues/9291

arg0 = jnp.ones((200,), dtype=jnp.int32)
arg1 = jnp.ones((1,), dtype=jnp.uint8)

# Attempting to replicate fails below.
class BugTest(iree.jax.Program):
  @iree.jax.kernel
  def _test(a0, a1):
    return jax.lax.dynamic_slice(a0, a1, (1,))

  def test(self, a0=iree.jax.like(arg0), a1=iree.jax.like(arg1)):
    return self._test(a0, a1)

ir = str(iree.jax.Program.get_mlir_module(BugTest()))
print(ir)


iree_config = iree.runtime.system_api.Config("local-task")
iree_binary = iree.compiler.compile_str(
    ir, target_backends=["cpu"], input_type="mhlo")
vm_module = iree.runtime.VmModule.from_flatbuffer(iree_binary)
module_object = iree.runtime.load_vm_module(vm_module, iree_config)
out = module_object["test"](np.arange(200, dtype=np.int32),
                            np.array((9,), np.uint8))
assert out.to_host()[0] == 9

