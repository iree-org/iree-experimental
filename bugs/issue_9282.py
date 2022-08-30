# RUN: %PYTHON %s
# XFAIL: *

import iree.compiler
import iree.jax
import iree.runtime
import jax.numpy as jnp

# https://github.com/iree-org/iree/issues/9282

arg0 = jnp.ones((5,), dtype=jnp.uint8)
arg1 = jnp.ones((5,), dtype=jnp.uint8)

# Attempting to replicate fails below.
class BugTest(iree.jax.Program):
  @iree.jax.kernel
  def _main(a0, a1):
    return a0 + a1

  def main(self, a0=iree.jax.like(arg0), a1=iree.jax.like(arg1)):
    return self._main(a0, a1)

ir = str(iree.jax.Program.get_mlir_module(BugTest()))
print(ir)


instance = iree.runtime.VmInstance()
iree_config = iree.runtime.system_api.Config("local-task")
iree_binary = iree.compiler.compile_str(
    ir, target_backends=["llvm-cpu"], input_type="mhlo")
vm_module = iree.runtime.VmModule.from_flatbuffer(instance, iree_binary)
module_object = iree.runtime.load_vm_module(vm_module, iree_config)
out = module_object["main"](arg0, arg1)
assert out.to_host().dtype == jnp.uint8

