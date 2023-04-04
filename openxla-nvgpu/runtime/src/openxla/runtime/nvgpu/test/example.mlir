// RUN: iree-compile %s --iree-hal-target-backends=cuda | openxla-runner - example.main | FileCheck %s

module @example {

  func.func private @cudnn.hello()

  func.func @main() {
    // CHECK: Hello from OpenXLA CuDNN Module!
    call @cudnn.hello() : () -> ()
    return
  }

}
