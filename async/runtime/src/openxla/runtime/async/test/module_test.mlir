// Tested by openxla/runtime/async/module_test.cpp.

vm.module public @async {
  vm.func private @await_delayed_token() -> i32 {
    %ref = vm.call @asynctest.return.delayed.token() : () -> !vm.ref<!async.value>
    vm.call @async.value.await(%ref) : (!vm.ref<!async.value>) -> ()
    %0 = vm.call @async.value.query(%ref) : (!vm.ref<!async.value>) -> i32
    vm.cond_br %0, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %c0 = vm.const.i32 42
    vm.return %c0 : i32
  ^bb2:  // pred: ^bb0
    vm.fail %0, "failed to wait on async token"
  }
  vm.export @await_delayed_token

  vm.func private @await_available_value() -> i32 {
    %ref = vm.call @asynctest.return.available.scalar() : () -> !vm.ref<!async.value>
    vm.call @async.value.await(%ref) : (!vm.ref<!async.value>) -> ()
    %0 = vm.call @async.value.query(%ref) : (!vm.ref<!async.value>) -> i32
    vm.cond_br %0, ^bb3(%0 : i32), ^bb1
    ^bb1:  // pred: ^bb0
      %1 = vm.call @async.value.load.i32(%ref) : (!vm.ref<!async.value>) -> i32
      %ref_0 = vm.call @asynctest.return.available.scalar() : () -> !vm.ref<!async.value>
      vm.call @async.value.await(%ref_0) : (!vm.ref<!async.value>) -> ()
      %2 = vm.call @async.value.query(%ref_0) : (!vm.ref<!async.value>) -> i32
      vm.cond_br %2, ^bb3(%2 : i32), ^bb2
    ^bb2:  // pred: ^bb1 
      %3 = vm.call @async.value.load.i32(%ref_0) : (!vm.ref<!async.value>) -> i32
      %4 = vm.add.i32 %1, %3 : i32
      vm.return %4 : i32
    ^bb3(%5: i32):  // 2 preds: ^bb0, ^bb1
      vm.fail %5, "failed to wait on async value"
  }
  vm.export @await_available_value

  vm.func private @await_delayed_value() -> i32 {
    %ref = vm.call @asynctest.return.delayed.scalar() : () -> !vm.ref<!async.value>
    vm.call @async.value.await(%ref) : (!vm.ref<!async.value>) -> ()
    %0 = vm.call @async.value.query(%ref) : (!vm.ref<!async.value>) -> i32
    vm.cond_br %0, ^bb3(%0 : i32), ^bb1
    ^bb1:  // pred: ^bb0
      %1 = vm.call @async.value.load.i32(%ref) : (!vm.ref<!async.value>) -> i32
      %ref_0 = vm.call @asynctest.return.delayed.scalar() : () -> !vm.ref<!async.value>
      vm.call @async.value.await(%ref_0) : (!vm.ref<!async.value>) -> ()
      %2 = vm.call @async.value.query(%ref_0) : (!vm.ref<!async.value>) -> i32
      vm.cond_br %2, ^bb3(%2 : i32), ^bb2
    ^bb2:  // pred: ^bb1 
      %3 = vm.call @async.value.load.i32(%ref_0) : (!vm.ref<!async.value>) -> i32
      %4 = vm.add.i32 %1, %3 : i32
      vm.return %4 : i32
    ^bb3(%5: i32):  // 2 preds: ^bb0, ^bb1
      vm.fail %5, "failed to wait on async value"
  }
  vm.export @await_delayed_value

  vm.func private @await_token_error() -> i32 {
    %ref = vm.call @asynctest.return.token.error() : () -> !vm.ref<!async.value>
    vm.call @async.value.await(%ref) : (!vm.ref<!async.value>) -> ()
    %0 = vm.call @async.value.query(%ref) : (!vm.ref<!async.value>) -> i32
    vm.cond_br %0, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0
    %c0 = vm.const.i32 42
    vm.return %c0 : i32
  ^bb2:  // pred: ^bb0
    vm.fail %0, "failed to wait on async token"
  }
  vm.export @await_token_error

  vm.func private @await_delayed_memref() -> !vm.ref<!hal.buffer_view> {
    %ref = vm.call @asynctest.return.delayed.memref() : () -> !vm.ref<!async.value>
    vm.call @async.value.await(%ref) : (!vm.ref<!async.value>) -> ()
    %0 = vm.call @async.value.query(%ref) : (!vm.ref<!async.value>) -> i32
    vm.cond_br %0, ^bb2, ^bb1
  ^bb1:  // pred: ^bb0    
    %1 = vm.call @async.value.load.ref(%ref) : (!vm.ref<!async.value>) -> !vm.ref<?>
    %2 = vm.cast.any.ref %1 : !vm.ref<?> -> !vm.ref<!hal.buffer_view>
    vm.return %2 : !vm.ref<!hal.buffer_view>
  ^bb2:  // pred: ^bb0
    vm.fail %0, "failed to wait on async value"
  }
  vm.export @await_delayed_memref

  vm.import private @async.value.await(!vm.ref<!async.value>)
  vm.import private @async.value.query(!vm.ref<!async.value>) -> i32
  vm.import private @async.value.load.i32(!vm.ref<!async.value>) -> i32
  vm.import private @async.value.load.ref(!vm.ref<!async.value>) -> !vm.ref<?>
  vm.import private @asynctest.return.available.scalar() -> !vm.ref<!async.value>
  vm.import private @asynctest.return.delayed.memref() -> !vm.ref<!async.value>
  vm.import private @asynctest.return.delayed.scalar() -> !vm.ref<!async.value>
  vm.import private @asynctest.return.delayed.token() -> !vm.ref<!async.value>
  vm.import private @asynctest.return.token.error() -> !vm.ref<!async.value>
}

