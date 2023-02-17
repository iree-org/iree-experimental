!A_size = tensor<${M}x${K}xf32>
!B_size = tensor<${K}x${N}xf32>
!C_size = tensor<${M}x${N}xf32>

func.func @matmul_static(
    %A : !A_size, %B : !B_size, %C : !C_size) -> !C_size {
  %0 = linalg.matmul ins(%A, %B : !A_size, !B_size)
                     outs(%C : !C_size) -> !C_size
  return %0 : !C_size
}
