// Instructions; TL;DR
// ===================
//
// This script shows a simple example of tiling for 2 levels and connecting to 
// wmma operations.
// This is purely for illustration purposes as this does not perform any 
// thread/warp level mapping or shared memory.
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
//   sed "s/\${M}/191/g" | sed "s/\${K}/1234/g" | sed "s/\${N}/511/g" | \
//   ${IREE_DIR}/build/tools/iree-opt \
//     --iree-hal-target-backends=llvm-cpu \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_DIR}/build/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))' \
//      --iree-codegen-llvmcpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cpu/matmul_step_04_tile_and_vector_mask.mlir \
//       --iree-codegen-llvmgpu-enable-transform-dialect-jit=false
// ```
//
// To execute:
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
//   sed "s/\${M}/191/g" | sed "s/\${K}/1234/g" | sed "s/\${N}/511/g" | \
//   iree-compile - --iree-hal-target-backends=llvm-cpu \
//     --iree-codegen-llvmcpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cpu/matmul_step_04_tile_and_vector_mask.mlir | \
//   iree-benchmark-module --batch-size=100 --benchmark-repetitions=100 --function=matmul_static \
//     --input="191x1234xf32=1" \
//     --input="1234x511xf32=2" \
//     --input="191x511xf32=3"
// ```
 
transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
 
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  %generic = transform.structured.generalize %matmul

  // Step 1. Tile to forall and sequential scf.for.
  // ======================================================
  %forall_l1, %generic_l1 =
    transform.iree.tile_to_forall_and_workgroup_count_region %generic
      num_threads [1]
      ( mapping = [#gpu.block<x>] )
  transform.iree.apply_patterns %variant_op {canonicalization, cse, tiling_canonicalization}
 
  // Step 2. Tile to sequential scf.for.
  // ======================================================
  %generic_l2, %loops_l1:3 = transform.structured.tile_to_scf_for %generic_l1 [6, 16, 1]
  transform.iree.apply_patterns %variant_op {canonicalization, cse, tiling_canonicalization}
 
  // Step 3. Vectorize.
  // ======================================================
  // Regular vectorization only kicks in when the tiled linalg op is static.
  //
  // %func = transform.structured.match ops{["func.func"]} in %variant_op
  //   : (!pdl.operation) -> !pdl.operation
  // %func_2 = transform.structured.vectorize %func
  //
  // Masked vectorization kicks in all the time.
  %func_2 = transform.structured.match ops{["func.func"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  transform.structured.masked_vectorize %generic_l2 vector_sizes [6, 16, 1]
  transform.iree.apply_patterns %variant_op {lower_vector_masks}
 
  // Post-vectorization canonicalizations and hoistings to avoid roundtripping 
  // vectors in memory and prepare for bufferization.
  transform.iree.apply_patterns %variant_op {canonicalization, cse, licm}
  // TODO: vector.transfer hoisting violates dominance in the presence of masks
  // if licm was not done explicitly before.
  %func_3 = transform.structured.hoist_redundant_tensor_subsets %func_2
    : (!pdl.operation) -> !pdl.operation
 
  // IREE-specific bufferization.
  // ============================================================================
  // Pre-buferization canonicalizations and cleanups help avoid extra copies.
  transform.iree.apply_patterns %variant_op {canonicalization, cse, licm}
  %variant_op_2 = transform.iree.bufferize %variant_op
  // %variant_op_2 = transform.iree.bufferize {test_analysis_only, print_conflicts} %variant_op
 
  // IREE-specific cleanup and connection to the runtime and threadpool, required
  // to run e2e.
  // ============================================================================
  %func_e = transform.structured.match ops{["func.func"]} in %variant_op_2
    : (!pdl.operation) -> !pdl.operation
  %func_e_2 = transform.iree.erase_hal_descriptor_type_from_memref %func_e
  %func_e_3 = transform.iree.forall_to_workgroup %func_e_2
  %func_e_4 = transform.iree.hoist_static_alloc %func_e_3
    : (!pdl.operation) -> !pdl.operation

  
  // Step 4. Late blanket/intrusive lowering of vector ops to vector abstractions
  // that are close to the LLVM level-of abstraction.
  // ============================================================================
  %func_e_5 = transform.vector.lower_vectors %func_e_4
    contraction_lowering = "outerproduct"
    transpose_lowering = "shuffle"
    // {unroll_vector_transfers = false}

  // TODO: maybe control transform.lower_to_llvm from here.
 
  // Late canonicalizations and cleanups.
  transform.iree.apply_patterns %variant_op_2 {canonicalization, cse, licm}
}
