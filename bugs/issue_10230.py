# RUN: %PYTHON %s
# XFAIL: *

import iree.compiler
import iree.jax
import iree.runtime
import jax
import jax.numpy as jnp

# https://github.com/iree-org/iree/issues/10230

arg0 = jnp.array([[3.0, 2.0, 4.0], [5, 6, 7], [3, 4, 5]])
arg1 = jnp.array([[3, 2, 4], [5, 6, 7], [3, 4, 5]])

class BugTest(iree.jax.Program):
  @iree.jax.kernel
  def _main(x, y):
    t = jnp.einsum("ki,jk,im,in,ix->ijmnx", x, jnp.einsum("ik,kj->ij", x, y) + jnp.power(x + y,3), y,x,y)
    return jax.nn.softmax(t)

  def main(self, x=iree.jax.like(arg0), y=iree.jax.like(arg1)):
    return self._main(x, y)

ir = str(iree.jax.Program.get_mlir_module(BugTest()))
print(ir)

iree_config = iree.runtime.system_api.Config("local-task")
iree_binary = iree.compiler.compile_str(
    ir, target_backends=["llvm-cpu"], input_type="mhlo")
vm_module = iree.runtime.VmModule.from_flatbuffer(instance, iree_binary)
module_object = iree.runtime.load_vm_module(vm_module, iree_config)
out = module_object["main"](arg0, arg1)
assert out.to_host().dtype == jnp.float64
