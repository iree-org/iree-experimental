// Instructions; TL;DR
// ===================
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/pointwise.mlir |\
//   sed "s/\${SZ1}/1024/g" | sed "s/\${SZ2}/1024/g" | \
//   sed "s/private @pointwise_1d_static(/@pointwise_1d_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   ${LLVM_BUILD_DIR}/bin/mlir-opt \
//     --pass-pipeline="builtin.module(test-transform-dialect-interpreter{\
//       transform-file-name=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/pointwise_mlir_pipeline_spec.mlir})" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt --promote-buffers-to-stack="max-rank-of-allocated-memref=2 max-alloc-size-in-bytes=10000000" \
//     --test-transform-dialect-erase-schedule \
//     --lower-affine --func-bufferize --canonicalize --cse --gpu-kernel-outlining --convert-scf-to-cf \
//     --expand-strided-metadata --convert-cf-to-llvm --convert-func-to-llvm --convert-vector-to-llvm \
//     --finalize-memref-to-llvm  --convert-index-to-llvm | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt \
//     --pass-pipeline='builtin.module(gpu.module(convert-gpu-to-nvvm,reconcile-unrealized-casts,test-gpu-to-cubin))'
// ```
//

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %generic = transform.structured.match ops{["linalg.generic"]} in %module_op
    : (!pdl.operation) -> !pdl.operation

  // Step 1. Parallelize.
  // ====================
  %forall_l1, %generic_l1 =
    transform.structured.tile_to_forall_op %generic tile_sizes [128]
      ( mapping = [#gpu.block<x>] )
  transform.structured.tile_to_forall_op %generic_l1 num_threads [32]
      ( mapping = [#gpu.linear<x>] )

  // Step 2. Vectorize.
  // ==================
  %func = transform.structured.match ops{["func.func"]} in %module_op
    : (!pdl.operation) -> !pdl.operation
  %func_2 = transform.structured.vectorize %func

  // Step 3. Bufferize.
  // ==================
  // Pre-bufferization canonicalizations and cleanups help avoid extra copies.
  // transform.iree.apply_patterns %variant_op {canonicalization, cse, licm}
  //   : (!pdl.operation) -> ()
  transform.bufferization.eliminate_empty_tensors %func_2
  %module_op_2 = transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %module_op 
    {bufferize_function_boundaries = true, allow_return_allocs = true, create_deallocs = false}
      : (!pdl.operation) -> !pdl.operation

  // Step 4. Map to blocks and threads.
  // ============================================================================
  %func_m = transform.structured.match ops{["func.func"]} in %module_op_2
    : (!pdl.operation) -> !pdl.operation
  %gpu_launch = transform.gpu.map_forall_to_blocks %func_m
    grid_dims = [256, 8] {generate_gpu_launch}
  transform.gpu.map_nested_forall_to_threads %gpu_launch 
    block_dims = [32, 4, 1]

  // Late canonicalizations and cleanups.
  // transform.iree.apply_patterns %module_op_2
  //   {canonicalization, cse, licm, tiling_canonicalization}
  //   : (!pdl.operation) -> ()


  %func_e = transform.structured.match ops{["func.func"]} in %module_op_2
    : (!pdl.operation) -> !pdl.operation
  %func_e_2 = transform.vector.transfer_to_scf %func_e
    max_transfer_rank = 1 full_unroll = true
      : (!pdl.operation) -> !pdl.operation
  %func_e_3 = transform.vector.lower_transfer %func_e_2
    max_transfer_rank = 1
      : (!pdl.operation) -> !pdl.operation
}
