// Example invocation:
// ===================
// export IREE_DIR=${HOME}/github/iree && \
// export IREE_SAMPLES_DIR=${HOME}/github/iree-samples && \
// M=123 && N=12345 && Mpadded=128 && Npadded=12352 && \
// cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/pad.mlir | \
// sed -e "s/\${M}/${M}/g" -e "s/\${N}/${N}/g" | \
// sed -e "s/\${Mpadded}/${Mpadded}/g" -e "s/\${Npadded}/${Npadded}/g" | \
// ${IREE_DIR}/build/tools/iree-compile - \
//   --iree-hal-benchmark-dispatch-repeat-count=5 \
//   --debug-only=transform-dialect-save-repro --mlir-disable-threading \
//   --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//   --iree-codegen-llvmgpu-enable-transform-dialect-pad-strategy \
//   --iree-flow-enable-pad-handling | \
// /usr/local/cuda/bin/nsys profile --stats=true \
// ${IREE_DIR}/build/tools/iree-run-module \
// --trace_execution=true \
// --module=- --function=foo --device=cuda \
// --input=${M}x${N}xf32=1 --input=${Mpadded}x${Npadded}xf32=1

// IREE performance without TD: drop --iree-codegen-llvmgpu-enable-transform-dialect-pad-strategy
// ==============================================================================================
// This mini-benchmark roughly shows: w/ TD ~9.5us w/o TD ~134.6us

// Example custom pad strategy: append the following.
// ==================================================
//   --td-pad-strategy-blk-sizes=8,8,1 | \
//   --td-pad-strategy-num-threads=4,4,1 | \
//   --td-pad-strategy-vector-size=2,2 | \

!tensor_t = tensor<${M}x${N}xf32>
!padded_tensor_t = tensor<${Mpadded}x${Npadded}xf32>

func.func @foo(%t : !tensor_t, %dummy_padded_t: !padded_tensor_t) -> !padded_tensor_t {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  %f0 = arith.constant 0.0 : f32
  %m = tensor.dim %t, %c0 : !tensor_t
  %n = tensor.dim %t, %c1 : !tensor_t
  %mp = tensor.dim %dummy_padded_t, %c0 : !padded_tensor_t
  %np = tensor.dim %dummy_padded_t, %c1 : !padded_tensor_t
  %h0 = arith.subi %mp, %m : index
  %h1 = arith.subi %np, %n : index
  %p = tensor.pad %t low[0,0] high[%h0, %h1] {
  ^bb0(%arg: index, %arg2: index):
    tensor.yield %f0 : f32
  } : !tensor_t to !padded_tensor_t
  return %p : !padded_tensor_t
}
