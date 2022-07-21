# RUN: %PYTHON %s
# XFAIL: *

import iree.compiler.tools.tflite as iree_tflite

# https://github.com/iree-org/iree/issues/9268
ir = '''
func.func @main(%a : tensor<f32>, %b : tensor<f32>) -> tensor<*xf32> {
  %val = "tfl.add"(%a, %b) {fused_activation_function = "NONE"} : (tensor<f32>, tensor<f32>) -> tensor<*xf32>
  return %val : tensor<*xf32>
}
'''
print(ir)
ir = iree_tflite.compile_str(ir, target_backends=["cpu"])
