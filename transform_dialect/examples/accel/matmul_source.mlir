func.func @matmul_example(%lhs: tensor<16x128xi8>, %rhs: tensor<128x128xi8>, %init : tensor<16x128xi32>) -> tensor<16x128xi32>
{
  %res = linalg.matmul ins(%lhs, %rhs: tensor<16x128xi8>, tensor<128x128xi8>)
                    outs(%init: tensor<16x128xi32>) -> tensor<16x128xi32>
  return %res : tensor<16x128xi32>
}
