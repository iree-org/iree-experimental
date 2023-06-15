// RUN: iree-opt %s --iree-plugin=openxla-async --split-input-file \
// RUN:   --async-to-async-runtime | \
// RUN: FileCheck %s

func.func @await_token(%arg0: !async.token){
  async.await %arg0 : !async.token
  return
}

// CHECK:  func.func @await_token(%[[ARG:.*]]: !async.value) {
// CHECK:    call @async.value.await(%[[ARG]]) : (!async.value) -> ()
// CHECK:    %[[R0:.*]] = call @async.value.query(%[[ARG]]) : (!async.value) -> i32
// CHECK:    util.status.check_ok %[[R0:.*]]
// CHECK:    return
// CHECK:  }
// CHECK:  func.func private @async.value.await(!async.value)

// -----

func.func @await_scalar_value(%arg0: !async.value<i32>) -> i32 {
  %0 = async.await %arg0 : !async.value<i32>
  return %0 : i32
}

// CHECK:  func.func @await_scalar_value(%[[ARG:.*]]: !async.value) -> i32 {
// CHECK:    call @async.value.await(%[[ARG]]) : (!async.value) -> ()
// CHECK:    %[[R0:.*]] = call @async.value.query(%[[ARG]]) : (!async.value) -> i32
// CHECK:    util.status.check_ok %[[R0:.*]]
// CHECK:    %[[R1:.*]] = call @async.value.load.i32(%[[ARG]]) : (!async.value) -> i32
// CHECK:    return %[[R1]] : i32
// CHECK:  }
// CHECK:  func.func private @async.value.await(!async.value)
// CHECK:  func.func private @async.value.query(!async.value) -> i32
// CHECK:  func.func private @async.value.load.i32(!async.value) -> i32

// -----

func.func @await_memref_value(%arg0: !async.value<memref<2xi32>>) -> memref<2xi32> {
  %0 = async.await %arg0 : !async.value<memref<2xi32>>
  return %0 : memref<2xi32>
}

// CHECK:  func.func @await_memref_value(%[[ARG:.*]]: !async.value) 
// CHECK-SAME:    -> memref<2xi32> {
// CHECK:    call @async.value.await(%[[ARG]]) : (!async.value) -> ()
// CHECK:    %[[R0:.*]] = call @async.value.load.ref(%[[ARG]]) : (!async.value) -> !util.object
// CHECK:    %[[R1:.*]] = util.cast %[[R0]] : !util.object to memref<2xi32>
// CHECK:    return %[[R1]] : memref<2xi32>
// CHECK:  }
// CHECK:  func.func private @async.value.load.ref(!async.value) -> !util.object
