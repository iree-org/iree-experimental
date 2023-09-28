func.func @matmul_example(%lhs: tensor<300x200xf32>, %rhs: tensor<200x100xf32>, %init : tensor<300x100xf32>) -> tensor<300x100xf32>
{
  %res = linalg.matmul ins(%lhs, %rhs: tensor<300x200xf32>, tensor<200x100xf32>)
                    outs(%init: tensor<300x100xf32>) -> tensor<300x100xf32>
  return %res : tensor<300x100xf32>
}