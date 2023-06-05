vm.module public @async {
  vm.func private @await_delayed_token() -> i32 {
    %ref = vm.call @asynctest.return.delayed.token() : () -> !vm.ref<!async.value>
    vm.call @async.value.await.value(%ref) : (!vm.ref<!async.value>) -> ()
    %c0 = vm.const.i32 42
    vm.return %c0 : i32
  }
  vm.export @await_delayed_token

  vm.func private @await_available_value() -> i32 {
    %ref = vm.call @asynctest.return.available.scalar() : () -> !vm.ref<!async.value>
    vm.call @async.value.await.value(%ref) : (!vm.ref<!async.value>) -> ()
    %0 = vm.call @async.value.load.i32(%ref) : (!vm.ref<!async.value>) -> i32
    %ref_0 = vm.call @asynctest.return.available.scalar() : () -> !vm.ref<!async.value>
    vm.call @async.value.await.value(%ref_0) : (!vm.ref<!async.value>) -> ()
    %1 = vm.call @async.value.load.i32(%ref_0) : (!vm.ref<!async.value>) -> i32
    %2 = vm.add.i32 %0, %1 : i32
    vm.return %2 : i32
  }
  vm.export @await_available_value

  vm.func private @await_delayed_value() -> i32 {
    %ref = vm.call @asynctest.return.delayed.scalar() : () -> !vm.ref<!async.value>
    vm.call @async.value.await.value(%ref) : (!vm.ref<!async.value>) -> ()
    %0 = vm.call @async.value.load.i32(%ref) : (!vm.ref<!async.value>) -> i32
    %ref_0 = vm.call @asynctest.return.delayed.scalar() : () -> !vm.ref<!async.value>
    vm.call @async.value.await.value(%ref_0) : (!vm.ref<!async.value>) -> ()
    %1 = vm.call @async.value.load.i32(%ref_0) : (!vm.ref<!async.value>) -> i32
    %2 = vm.add.i32 %0, %1 : i32
    vm.return %2 : i32
  }
  vm.export @await_delayed_value

  vm.import private @async.value.await.value(!vm.ref<!async.value>)
  vm.import private @async.value.load.i32(!vm.ref<!async.value>) -> i32
  vm.import private @asynctest.return.delayed.token() -> !vm.ref<!async.value>
  vm.import private @asynctest.return.available.scalar() -> !vm.ref<!async.value>
  vm.import private @asynctest.return.delayed.scalar() -> !vm.ref<!async.value>
}

