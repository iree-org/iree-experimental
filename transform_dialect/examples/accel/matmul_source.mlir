func.func @matmul_example(%lhs: tensor<16x128xf32>, %rhs: tensor<128x128xf32>, %init : tensor<16x128xf32>) -> tensor<16x128xf32>
{
  %res = linalg.matmul ins(%lhs, %rhs: tensor<16x128xf32>, tensor<128x128xf32>)
                    outs(%init: tensor<16x128xf32>) -> tensor<16x128xf32>
  return %res : tensor<16x128xf32>
}
