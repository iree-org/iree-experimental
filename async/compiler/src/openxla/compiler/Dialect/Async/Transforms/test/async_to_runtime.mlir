// RUN: iree-opt --iree-plugin=openxla-async --split-input-file \
// RUN:   --async-to-async-runtime | \
// RUN: FileCheck %s

// CHECK-LABEL: @await_token
func.func @await_token(%arg0: !async.token){
  // CHECK: call @async.token.await(%arg0)
  async.await %arg0 : !async.token
  return
}

// -----

// CHECK-LABEL: @await_scalar_value
func.func @await_scalar_value(%arg0: !async.value<i32>) -> i32 {
  // CHECK: async.value.await.i32 %arg0
  %0 = async.await %arg0 : !async.value<i32>
  return %0 : i32
}

// -----

func.func @await_memref_value(%arg0: !async.value<memref<2xi32>>) -> memref<2xi32> {
  %0 = async.await %arg0 : !async.value<memref<2xi32>>
  return %0 : memref<2xi32>
}
// func.func @await_value(%arg0: !async.value) -> memref<2xi32> {
//   %1 = call async.await.value.ref(%0 : !async.value) -> !util.object
//   %2 = util.cast %1 : !util.object to !memref<2xi32>
//   return %2 : !memref<2xi32>
// }