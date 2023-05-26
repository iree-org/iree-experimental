// RUN: iree-compile %s --compile-to=vm --iree-execution-model=host-only | openxla-runner - token.main | FileCheck %s

module {
  func.func @await_token(%arg0: !async.value) {
    call @async.value.await.token(%arg0) : (!async.value) -> ()
    return
  }
  func.func private @async.value.await.token(!async.value)
}
