// Instructions; TL;DR
// ===================
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/pointwise.mlir |\
//   sed "s/\${SZ1}/4096/g" | sed "s/\${SZ2}/1/g" | \
//   sed "s/private @pointwise_1d_static(/@pointwise_1d_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   ${IREE_DIR}/build/tools/iree-opt \
//     --iree-hal-target-backends=cuda \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_DIR}/build/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
//      --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/pointwise_iree_pipeline_spec.mlir \
//      --iree-codegen-llvmgpu-enable-transform-dialect-jit=false
/// ```
//

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %generic = transform.structured.match ops{["linalg.generic"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  // Step 1. Parallelize.
  // ====================
  %forall_l1, %generic_l1 =
    transform.iree.tile_to_forall_and_workgroup_count_region %generic tile_sizes [128]
      ( mapping = [#gpu.block<x>] )
  transform.structured.tile_to_forall_op %generic_l1 num_threads [32]
      ( mapping = [#gpu.linear<x>] )

  // Step 2. Vectorize.
  // ==================
  %func = transform.structured.match ops{["func.func"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  transform.structured.vectorize %func

  // Step 3. Bufferize.
  // ==================
  // Pre-bufferization canonicalizations and cleanups help avoid extra copies.
  transform.iree.apply_patterns %variant_op {canonicalization, cse, licm}
    : (!pdl.operation) -> ()
  transform.iree.eliminate_empty_tensors %variant_op : (!pdl.operation) -> ()
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op
    : (!pdl.operation) -> (!pdl.operation)
  %func_m = transform.structured.match ops{["func.func"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %func_m
    : (!pdl.operation) -> ()

  // Step 4. Map to blocks and threads.
  // ============================================================================
  transform.iree.apply_patterns %variant_op_3 
    {canonicalization, cse, licm, tiling_canonicalization}
    : (!pdl.operation) -> ()
  transform.iree.forall_to_workgroup %func_m : (!pdl.operation) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_m
      workgroup_dims = [32, 1, 1]
    : (!pdl.operation) -> ()

  // Late canonicalizations and cleanups.
  transform.iree.apply_patterns %variant_op_3
    {canonicalization, cse, licm, tiling_canonicalization}
    : (!pdl.operation) -> ()
}
