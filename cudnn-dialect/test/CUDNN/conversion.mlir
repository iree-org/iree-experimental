// RUN: cudnn-opt --convert-cudnn-to-llvm %s | FileCheck %s

// CHECK: llvm.mlir.global

func.func @graph(%handle: !cudnn.handle,
      %input : tensor<4x24x31x31xf16>, %filter : tensor<32x24x3x3xf16>) ->
    tensor<4x24x31x31xf16> {

  %ret = cudnn.build_and_exec_graph %handle (%input, %filter) {
    ^bb0(
      %h: !cudnn.handle,
      %inp : !cudnn.tensor_desc<4x24x31x31xf16, alignment=16, stride=[23064, 1, 744, 24]>,
      %f : !cudnn.tensor_desc<32x24x3x3xf16, alignment=16, stride=[216, 1, 72, 24]>):
    %relu = cudnn.pointwise_relu(%inp) type=f16 lower_clip=0.0 :
      !cudnn.tensor_desc<4x24x31x31xf16, alignment=16,stride=[23064, 1, 744, 24]> ->
      !cudnn.tensor_desc<4x24x31x31xf16, alignment=16, stride=[23064, 1, 744, 24]>
  // %y = cudnn.convolution(%relu, %filter) type=f16 alpha=1.0 beta=0.0 spatial_dim_count=2
  //   spatial_stride=[1,1] pre_padding=[1,1] post_padding=[1,1] dilation=[1,1] :
  //     !cudnn.tensor_desc<4x24x31x31xf16, alignment=16,stride=[23064, 1, 744, 24]>,
  //     !cudnn.tensor_desc<32x24x3x3xf16, alignment=16, stride=[216,1,72,24]> ->
  //        !cudnn.tensor_desc<4x32x31x31xf16, alignment=16, stride=[30752, 1, 992, 32]>
    cudnn.build_graph(%relu) :
      !cudnn.tensor_desc<4x24x31x31xf16, alignment=16, stride=[23064, 1, 744, 24]>
  } : tensor<4x24x31x31xf16>, tensor<32x24x3x3xf16> -> tensor<4x24x31x31xf16>

  return %ret:  tensor<4x24x31x31xf16>
}
