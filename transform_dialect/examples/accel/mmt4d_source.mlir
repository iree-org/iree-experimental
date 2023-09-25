func.func @mmt4d_384x384x512_8x1x8() -> tensor<48x64x8x8xf32> {
    %lhs = util.unfoldable_constant dense<1.0> : tensor<48x384x8x1xf32>
    %rhs = util.unfoldable_constant dense<1.0> : tensor<64x384x8x1xf32>
    %dst = util.unfoldable_constant dense<1.0> : tensor<48x64x8x8xf32>
    %0 = linalg.mmt4d ins(%lhs, %rhs : tensor<48x384x8x1xf32>, tensor<64x384x8x1xf32>) outs(%dst : tensor<48x64x8x8xf32>) -> tensor<48x64x8x8xf32>
    return %0 : tensor<48x64x8x8xf32>
}
