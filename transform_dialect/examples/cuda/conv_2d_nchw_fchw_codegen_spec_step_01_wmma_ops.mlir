// Instructions; TL;DR
// ===================
//
// This script shows a simple example of tiling for 2 levels and connecting to 
// wmma operations.
// This is purely for illustration purposes as this does not perform any 
// thread/warp level mapping or shared memory.
//
// ```
//    export IREE_DIR=${HOME}/github/iree; \
//    export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//    cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/conv_2d_nchw_fchw.mlir |\
//    sed "s/\${N}/16/g" | sed "s/\${C}/16/g" | sed "s/\${F}/64/g" | \
//    sed "s/\${H}/132/g" | sed "s/\${W}/132/g" | \
//    sed "s/\${KH}/3/g" | sed "s/\${KW}/3/g" | \
//    sed "s/\${OH}/130/g" | sed "s/\${OW}/130/g" | \
//    ${IREE_DIR}/build/tools/iree-opt \
//      --iree-hal-target-backends=cuda \
//      --iree-abi-transformation-pipeline \
//      --iree-flow-transformation-pipeline \
//      --iree-stream-transformation-pipeline \
//      --iree-hal-configuration-pipeline | \
//    ${IREE_DIR}/build/tools/iree-opt \
//       --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
//       --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/conv_2d_nchw_fchw_codegen_spec_step_01_wmma_ops.mlir \
//       --iree-codegen-llvmgpu-enable-transform-dialect-jit=false 
// ```

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %named_conv = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  %conv = transform.structured.generalize %named_conv

  // Step 1. Tile to forall and sequential scf.for.
  // ======================================================
  %forall_l1, %conv_l1 =
    transform.iree.tile_to_forall_and_workgroup_count_region %conv 
      //          N, F, OH, OW, C, KH, KW
      tile_sizes [0, 0,  6,  6]
      ( mapping = [#gpu.block<y>, #gpu.block<x>] )
  %conv_l2, %loops:7 = transform.structured.tile_to_scf_for %conv_l1 
      //          N,  F, OH, OW, C, KH, KW
                [16, 16,  1,  1, 16,  1,  1]

  // Step 2. Pad the matmul and force packing to create the buffer in shared memory
  // Note: hoisting here may be dangerous memory-consumption-wise and we may be
  // better off with pipelining only.
  // ==============================================================================
  %conv_l2_padded = transform.structured.pad %conv_l2 {
    padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32], 
    padding_dimensions = [0, 1, 2], 
    pack_paddings=[1, 1, 1]
    , hoist_paddings=[2, 1, 0]
  }

  // Step 3. Rewrite tensor.pad in DPS. 
  // TODO: This must introduce unfoldable copies that disable the 
  // tensor::InsertSliceOp::fold very aggressive blanket behavior
  // ==============================================================
  %pad = transform.structured.match ops{["tensor.pad"]} in %variant_op 
    : (!pdl.operation) -> !pdl.operation
  %padded = transform.structured.rewrite_in_destination_passing_style %pad 
    : (!pdl.operation) -> !pdl.operation

  // Step 4. Map to threads, **SIMT** programming model.
  // Ensure we lower to 1-D vectors otherwise cp.async will not kick in.
  // ===================================================================
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  transform.structured.tile_to_forall_op %fill num_threads [16, 16]
      ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )
  %copy = transform.structured.match ops{["linalg.copy"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  transform.structured.tile_to_forall_op %copy num_threads [16, 16]
      ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )

  // Step 5. Rank-reduce and vectorize.
  // ==================================
  %func_v = transform.structured.match ops{["func.func"]} in %variant_op 
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func_v { rank_reducing_linalg, rank_reducing_vector }
    : (!pdl.operation) -> ()
  %func_v_3 = transform.structured.vectorize %func_v
  transform.iree.apply_patterns %func_v_3 { unroll_vectors_gpu_wmma }
    : (!pdl.operation) -> ()
  %func_v_4 = transform.structured.hoist_redundant_tensor_subsets %func_v_3
    : (!pdl.operation) -> !pdl.operation

  // Step 6. Bufferize and drop HAL descriptor from memref ops.
  // ==========================================================
  transform.iree.apply_patterns %func_v_4 {canonicalization, cse, licm }
    : (!pdl.operation) -> ()
  transform.iree.eliminate_empty_tensors %variant_op : (!pdl.operation) -> ()
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op
    : (!pdl.operation) -> (!pdl.operation)
  %func_m = transform.structured.match ops{["func.func"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %func_m
      : (!pdl.operation) -> ()

  // Step 7. Post-bufferization mapping workgroup.
  // =============================================
  transform.iree.forall_to_workgroup %func_m
    : (!pdl.operation) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_m
    workgroup_dims = [16, 16, 1]
      : (!pdl.operation) -> ()
  transform.iree.apply_buffer_optimizations %func_m
    : (!pdl.operation) -> ()

  // This must occur after bufferization because of the fancy CUDA types.
  transform.iree.apply_patterns %func_m {canonicalization, cse, licm, fold_memref_aliases }
    : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func_m { unroll_vectors_gpu_wmma }
    : (!pdl.operation) -> ()
  // TODO: Atm the slice does not get rewritten, seems brittle, figure this out.
  transform.iree.vector.vector_to_mma_conversion %func_m { use_wmma }
      : (!pdl.operation) -> ()
}
