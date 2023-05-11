// Instructions; TL;DR
// ===================
//
// This script shows a simple example of a convwith 1 dims that reduces to a 
// matmul along with a vanilla tiling and vectorization strategy.
//
// ```
//    export IREE_DIR=${HOME}/github/iree; \
//    export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//    cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/conv_2d_nhwc_hwcf.mlir |\
//    sed "s/\${N}/1/g" | sed "s/\${C}/1/g" | sed "s/\${F}/128/g" | \
//    sed "s/\${H}/41/g" | sed "s/\${W}/140/g" | \
//    sed "s/\${KH}/1/g" | sed "s/\${KW}/140/g" | \
//    sed "s/\${OH}/41/g" | sed "s/\${OW}/1/g" | \
//    ${IREE_DIR}/build/tools/iree-opt \
//      --iree-hal-target-backends=cuda \
//      --iree-abi-transformation-pipeline \
//      --iree-flow-transformation-pipeline \
//      --iree-stream-transformation-pipeline \
//      --iree-hal-configuration-pipeline | \
//    ${IREE_DIR}/build/tools/iree-opt \
//       --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
//       --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cpu/conv_2d_nhwc_hwcf_codegen_spec.mlir \
//       --iree-codegen-llvmgpu-enable-transform-dialect-jit=false
// ```
//
// To execute:
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/conv_2d_nhwc_hwcf.mlir |\
//   sed "s/\${N}/1/g" | sed "s/\${C}/1/g" | sed "s/\${F}/128/g" | \
//   sed "s/\${H}/41/g" | sed "s/\${W}/140/g" | \
//   sed "s/\${KH}/1/g" | sed "s/\${KW}/140/g" | \
//   sed "s/\${OH}/41/g" | sed "s/\${OW}/1/g" | \
//   iree-compile - --iree-hal-target-backends=llvm-cpu \
//     --iree-codegen-llvmcpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cpu/conv_2d_nhwc_hwcf_codegen_spec.mlir | \
//   iree-benchmark-module --batch-size=10 --benchmark-repetitions=10  --function=conv \
//     --input="1x41x140x1xf32=1" \
//     --input="1x140x1x128xf32=2" \
//     --input="1x41x1x128xf32=3"
// ```
// 
// Prints: 
// BM_conv/process_time/real_time 0.021 ms 0.023 ms 35154 items_per_second=47.7327k/s

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):

  %named_conv = transform.structured.match ops{["linalg.conv_2d_nhwc_hwcf"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  
  %conv = transform.structured.generalize %named_conv

  // Turns out the 1x1 ... conv is actually just a matmul.
  %func = transform.structured.match ops{["func.func"]} in %variant_op 
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func { rank_reducing_linalg }
    : (!pdl.operation) -> ()

  // TODO: Add a transform.structured.specialize that can match a few different ops
  // Then, this reduces to just a linalg.matmul and we can reuse existing strategies.

  // Step 1. Tile to forall and sequential scf.for.
  // ======================================================
  %forall_l1, %conv_l1 =
    transform.structured.tile_to_forall_op %conv
      tile_sizes [1]
      ( mapping = [#gpu.block<x>] )
  transform.iree.populate_workgroup_count_region_using_num_threads_slice
    %forall_l1 : (!pdl.operation) -> ()

  // Step 2. Tile to sequential scf.for.
  // ======================================================
  %matmul_l2, %loops_l1:3 = transform.structured.tile_to_scf_for %conv_l1 [1, 4, 4]
  transform.iree.apply_patterns %func { rank_reducing_linalg }
    : (!pdl.operation) -> ()
  
  // Step 3. Vectorize.
  // ======================================================
  %func_v = transform.structured.match ops{["func.func"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func_v { rank_reducing_linalg }
    : (!pdl.operation) -> ()
  %func_v_2 = transform.structured.vectorize %func_v
  // Post-vectorization canonicalizations and hoistings to avoid roundtripping 
  // vectors in memory and prepare for bufferization.
  // TODO: some weirdness here with orderings, split in 2to better control for now.
  transform.iree.apply_patterns %variant_op {canonicalization, cse}
    : (!pdl.operation) -> ()
  transform.iree.apply_patterns %variant_op {fold_tensor_subsets}
    : (!pdl.operation) -> ()
  transform.structured.hoist_redundant_tensor_subsets %func_v_2
    : (!pdl.operation) -> !pdl.operation

  // IREE-specific bufferization.
  // ============================================================================
  // Pre-buferization canonicalizations and cleanups help avoid extra copies.
  transform.iree.apply_patterns %variant_op {canonicalization, cse, licm}
    : (!pdl.operation) -> ()
  %variant_op_2 = transform.iree.bufferize %variant_op
    : (!pdl.operation) -> !pdl.operation
  // This is to debug bufferization if needed.
  // %variant_op_2 = transform.iree.bufferize {test_analysis_only, print_conflicts} %variant_op

  // IREE-specific cleanup and connection to the runtime and threadpool, required
  // to run e2e.
  // ============================================================================
  %func_e = transform.structured.match ops{["func.func"]} in %variant_op_2
    : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %func_e
    : (!pdl.operation) -> ()
  transform.iree.forall_to_workgroup %func_e
    : (!pdl.operation) -> ()
  
  // Step 4. Late blanket/intrusive lowering of vector ops to vector abstractions
  // that are close to the LLVM level-of abstraction.
  // ============================================================================
  // TODO: some weirdness happening here that needs to be investigated.
  // %func_e_4 = transform.vector.lower_vectors %func_e
  //   contraction_lowering = "outerproduct"
  //   transpose_lowering = "shuffle"
    // {unroll_vector_transfers = false}
    // : (!pdl.operation) -> !pdl.operation

  // Late canonicalizations and cleanups.
  transform.iree.apply_patterns %variant_op_2 {canonicalization, cse, licm}
    : (!pdl.operation) -> ()
}
