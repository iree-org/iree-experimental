// RUN: iree-compile %s --iree-execution-model=host-only | openxla-runner - async.main | FileCheck %s

module @async{
  func.func @main() -> i32 {
    %0 = call @async.test.value() : () -> !async.value
    %1 = call @async.value.await.i32(%0) : (!async.value) -> i32
    %2 = call @async.test.value() : () -> !async.value
    %3 = call @async.value.await.i32(%2) : (!async.value) -> i32
    %4 = arith.addi %1, %3 : i32
    return %4 : i32
  }
  func.func private @async.value.await.i32(!async.value) -> i32
  func.func private @async.test.value() ->!async.value
}
