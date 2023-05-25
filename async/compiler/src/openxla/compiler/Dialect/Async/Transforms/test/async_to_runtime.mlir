// RUN: iree-opt --iree-plugin=openxla-async --split-input-file \
// RUN:   --async-to-async-runtime | \
// RUN: FileCheck %s

func.func @await_token(%arg0: !async.token){
  // CHECK: call @async.value.await.token
  async.await %arg0 : !async.token
  return
}

// CHECK:  func.func @await_token(%[[ARG:.*]]: !async.value) {
// CHECK:     call @async.value.await.token(%[[ARG]]) : (!async.value) -> ()
// CHECK:     return
// CHECK:   }
// CHECK:   func.func private @async.value.await.token(!async.value)

// -----

func.func @await_scalar_value(%arg0: !async.value<i32>) -> i32 {
  %0 = async.await %arg0 : !async.value<i32>
  return %0 : i32
}

// CHECK:    func.func @await_scalar_value(%[[ARG:.*]]: !async.value) -> i32 {
// CHECK:      %[[R0:.*]] = call @async.value.await.i32(%[[ARG]]) : (!async.value) -> i32
// CHECK:      return %[[R0]] : i32
// CHECK:    }
// CHECK:    func.func private @async.value.await.i32(!async.value) -> i32

// -----

func.func @await_memref_value(%arg0: !async.value<memref<2xi32>>) -> memref<2xi32> {
  %0 = async.await %arg0 : !async.value<memref<2xi32>>
  return %0 : memref<2xi32>
}

// CHECK:  func.func @await_memref_value(%[[ARG:.*]]: !async.value) 
// CHECK:    -> memref<2xi32> {
// CHECK:    %[[R0:.*]] = call @async.value.await.ref(%[[ARG]]) : (!async.value) -> !util.object
// CHECK:    %[[R1:.*]] = util.cast %[[R0]] : !util.object to memref<2xi32>
// CHECK:    return %[[R1]] : memref<2xi32>
// CHECK:  }
// CHECK:  func.func private @async.value.await.ref(!async.value) -> !util.object