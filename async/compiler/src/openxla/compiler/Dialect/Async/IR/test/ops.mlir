// RUN: iree-opt --iree-plugin=openxla-async --split-input-file %s \
// RUN:   | FileCheck %s

// CHECK-LABEL: @identity_token
func.func @identity_token(%arg0: !async.token) -> !async.token {
  // CHECK: return %arg0 : !async.token
  return %arg0 : !async.token
}

// CHECK-LABEL: @identity_value
func.func @identity_value(%arg0 : !async.value<f32>) -> !async.value<f32> {
  // CHECK: return %arg0 : !async.value<f32>
  return %arg0 : !async.value<f32>
}

// CHECK-LABEL: @await_token
func.func @await_token(%arg0: !async.token) {
  // CHECK: async.await %arg0
  async.await %arg0 : !async.token
  return
}

// CHECK-LABEL: @await_value
func.func @await_value(%arg0: !async.value<f32>) -> f32 {
  // CHECK: async.await %arg0
  %0 = async.await %arg0 : !async.value<f32>
  return %0 : f32
}