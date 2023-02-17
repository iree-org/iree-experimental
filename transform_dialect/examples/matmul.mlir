!A_size = tensor<1024x2048xf32>
!B_size = tensor<2048x4096xf32>
!C_size = tensor<1024x4096xf32>

func.func @matmul_static(
    %A : !A_size, %B : !B_size, %C : !C_size) -> !C_size {
  %0 = linalg.matmul ins(%A, %B : !A_size, !B_size)
                     outs(%C : !C_size) -> !C_size
  return %0 : !C_size
}
