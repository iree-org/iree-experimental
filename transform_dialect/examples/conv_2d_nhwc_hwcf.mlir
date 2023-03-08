!input_tensor_t = tensor<${N}x${H}x${W}x${C}xf32>
!weight_tensor_t = tensor<${KH}x${KW}x${C}x${F}xf32>
!output_tensor_t = tensor<${N}x${OH}x${OW}x${F}xf32>

func.func @conv(%in: !input_tensor_t, %ker: !weight_tensor_t,
                %out: !output_tensor_t) -> !output_tensor_t {
  // %empty = tensor.empty() :!output_tensor_t

  // %c0 = arith.constant 0.000000e+00 : f32
  // %fill = linalg.fill ins(%c0 : f32) outs(%empty: !output_tensor_t) ->!output_tensor_t

  %conv = linalg.conv_2d_nhwc_hwcf
    {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>}
    ins(%in, %ker : !input_tensor_t, !weight_tensor_t)
    outs(%out :!output_tensor_t) ->!output_tensor_t

 func.return %conv :!output_tensor_t
}
