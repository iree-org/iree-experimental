// RUN: iree-opt %s --iree-plugin=openxla_nvgpu --pass-pipeline='builtin.module(openxla-nvgpu-convert-mhlo-to-cudnn)' | FileCheck %s

!tensor = tensor<1x16x32x8xf32>

func.func @conv(%argument: !tensor) -> !tensor {
  %min = mhlo.constant dense<0.0> : !tensor
  %max = mhlo.constant dense<0xFFFFFFFF> : !tensor
  // CHECK: cudnn.pointwise_relu(%0) type = f32 lower_clip = 0.000000e+00
  // CHECK-SAME: -> !cudnn.tensor_desc<
  // CHECK-SAME:      1x16x32x8xf32,
  // CHECK-SAME:      alignment = 0,
  // CHECK-SAME:      stride = [4096, 256, 8, 1]
  // CHECK-SAME:    >
  %result = mhlo.clamp %min, %argument, %max : !tensor
  return %result : !tensor
}