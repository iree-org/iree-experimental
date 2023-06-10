// Example invocation:
// ===================
// export IREE_DIR=${HOME}/github/iree && \
// export IREE_SAMPLES_DIR=${HOME}/github/iree-samples && \
// M=123 && N=456 && K=12345 && FUN_NAME=fill_matmul_static && \
// cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir | \
// sed -e "s/\${M}/${M}/g" -e "s/\${N}/${N}/g" -e "s/\${K}/${K}/g" | \
// sed "s/private @${FUN_NAME}(/@${FUN_NAME}(/g" | \
// ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
// ${IREE_DIR}/build/tools/iree-compile - \
//   --iree-hal-benchmark-dispatch-repeat-count=5 \
//   --debug-only=transform-dialect-save-repro --mlir-disable-threading \
//   --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 | \
// /usr/local/cuda/bin/nsys profile --stats=true \
// ${IREE_DIR}/build/tools/iree-run-module \
// --trace_execution=true \
// --module=- --function=${FUN_NAME} --device=cuda \
// --input=${M}x${K}xf32=1 --input=${K}x${N}xf32=1 --input=${M}x${N}xf32=0

!A_t = tensor<${M}x${K}xf32>
!B_t = tensor<${K}x${N}xf32>
!C_t = tensor<${M}x${N}xf32>

func.func private @matmul_static(
    %A : !A_t, %B : !B_t, %C : !C_t) -> !C_t {
  %0 = linalg.matmul ins(%A, %B : !A_t, !B_t)
                     outs(%C : !C_t) -> !C_t
  return %0 : !C_t
}

func.func private @fill_matmul_static(
    %A : !A_t, %B : !B_t, %C : !C_t) -> !C_t {
  %f0 = arith.constant 0.0 : f32
  %out = tensor.empty() : !C_t
  %filled = linalg.fill ins(%f0 : f32) outs(%out : !C_t) -> !C_t
  %0 = linalg.matmul ins(%A, %B : !A_t, !B_t)
                     outs(%filled : !C_t) -> !C_t
  return %0 : !C_t
}

!A_dyn_t = tensor<?x?xf32>
!B_dyn_t = tensor<?x?xf32>
!C_dyn_t = tensor<?x?xf32>

func.func private @matmul_dynamic(
    %A : !A_dyn_t, %B : !B_dyn_t, %C : !C_dyn_t) -> !C_dyn_t {
  %0 = linalg.matmul ins(%A, %B : !A_dyn_t, !B_dyn_t)
                     outs(%C : !C_dyn_t) -> !C_dyn_t
  return %0 : !C_dyn_t
}

func.func private @fill_matmul_dynamic(
    %A : !A_dyn_t, %B : !B_dyn_t, %C : !C_dyn_t) -> !C_dyn_t {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %d0 = tensor.dim %C, %c0 : !C_dyn_t
  %d1 = tensor.dim %C, %c1 : !C_dyn_t
  %out = tensor.empty(%d0, %d1) : !C_dyn_t
  %filled = linalg.fill ins(%f0 : f32) outs(%out : !C_dyn_t) -> !C_dyn_t
  %0 = linalg.matmul ins(%A, %B : !A_dyn_t, !B_dyn_t)
                     outs(%filled : !C_dyn_t) -> !C_dyn_t
  return %0 : !C_dyn_t
}
