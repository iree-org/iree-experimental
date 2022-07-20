# RUN: %PYTHON %s
# XFAIL: *

import iree.compiler
import iree.jax
import jax.lax
import jax.numpy as jnp

# https://github.com/iree-org/iree/issues/9669

arg0 = jnp.ones((30522, 17), dtype=jnp.float32)
arg1 = jnp.ones((512), dtype=jnp.int32)
arg2 = jnp.ones((512, 17), dtype=jnp.float32)

# Attempting to replicate fails below.
class BugTest(iree.jax.Program):
  @iree.jax.kernel
  def _test(a0, a1, a2):
    return a0.at[a1].set(a2)

  def test(self, a0=iree.jax.like(arg0), a1=iree.jax.like(arg1), a2=iree.jax.like(arg2)):
    return self._test(a0, a1, a2)

ir = '''
func.func @main(%arg0: tensor<30522x384xf32>, %arg1: tensor<512xi32>, %arg2: tensor<512x384xf32>) -> tensor<30522x384xf32> {
  %0 = "mhlo.scatter"(%arg0, %arg1, %arg2) ({
  ^bb0(%arg3: tensor<f32>, %arg4: tensor<f32>):
    %1 = mhlo.add %arg3, %arg4 : tensor<f32>
    "mhlo.return"(%1) : (tensor<f32>) -> ()
  }) {indices_are_sorted = false, scatter_dimension_numbers = #mhlo.scatter<update_window_dims = [1], inserted_window_dims = [0], scatter_dims_to_operand_dims = [0], index_vector_dim = 1>, unique_indices = false} : (tensor<30522x384xf32>, tensor<512xi32>, tensor<512x384xf32>) -> tensor<30522x384xf32>
  return %0 : tensor<30522x384xf32>
}
'''


print(ir)
# ir = str(iree.jax.Program.get_mlir_module(BugTest()))
# print(ir)

# Compilation fails.
iree.compiler.compile_str(ir, target_backends=["cpu"], input_type="mhlo")