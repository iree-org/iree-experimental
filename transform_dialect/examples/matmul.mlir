!A_size = tensor<${M}x${K}xf32>
!B_size = tensor<${K}x${N}xf32>
!C_size = tensor<${M}x${N}xf32>

func.func private @matmul_static(
    %A : !A_size, %B : !B_size, %C : !C_size) -> !C_size {
  %0 = linalg.matmul ins(%A, %B : !A_size, !B_size)
                     outs(%C : !C_size) -> !C_size
  return %0 : !C_size
}

func.func private @fill_matmul_static(
    %A : !A_size, %B : !B_size, %C : !C_size) -> !C_size {
  %f0 = arith.constant 0.0 : f32
  %out = tensor.empty() : !C_size
  %filled = linalg.fill ins(%f0 : f32) outs(%out : !C_size) -> !C_size
  %0 = linalg.matmul ins(%A, %B : !A_size, !B_size)
                     outs(%filled : !C_size) -> !C_size
  return %0 : !C_size
}
