// Instructions; TL;DR
// ===================
//
// This script shows a simple example of a conv_2d_nchw_fchw that reduces to a 
// conv_1d_ncw_fcw along with a tiling, padding and vectorization strategy.
//
// ```
//    export IREE_DIR=${HOME}/github/iree; \
//    export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//    cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/conv_2d_nchw_fchw.mlir |\
//    sed "s/\${N}/16/g" | sed "s/\${C}/128/g" | sed "s/\${F}/256/g" | \
//    sed "s/\${H}/42/g" | sed "s/\${W}/68/g" | \
//    sed "s/\${KH}/3/g" | sed "s/\${KW}/3/g" | \
//    sed "s/\${OH}/40/g" | sed "s/\${OW}/66/g" | \
//    ${IREE_DIR}/build/tools/iree-opt \
//      --iree-hal-target-backends=cuda \
//      --iree-abi-transformation-pipeline \
//      --iree-flow-transformation-pipeline \
//      --iree-stream-transformation-pipeline \
//      --iree-hal-configuration-pipeline | \
//    ${IREE_DIR}/build/tools/iree-opt \
//       --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
//       --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cpu/conv_2d_nchw_fchw_codegen_spec.mlir \
//       --iree-codegen-llvmgpu-enable-transform-dialect-jit=false
// ```
//
// To execute:
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/conv_2d_nchw_fchw.mlir |\
//    sed "s/\${N}/4/g" | sed "s/\${C}/128/g" | sed "s/\${F}/256/g" | \
//    sed "s/\${H}/42/g" | sed "s/\${W}/68/g" | \
//    sed "s/\${KH}/3/g" | sed "s/\${KW}/3/g" | \
//    sed "s/\${OH}/40/g" | sed "s/\${OW}/66/g" | \
//   iree-compile - --iree-hal-target-backends=llvm-cpu \
//     --iree-codegen-llvmcpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cpu/conv_2d_nchw_fchw_codegen_spec.mlir | \
//    iree-benchmark-module --batch-size=10 --benchmark-repetitions=10  --function=conv_2d_nchw_fchw \
//     --input="4x128x42x68xf32=1" \
//     --input="256x128x3x3xf32=2" \
//     --input="4x256x40x66xf32=3"
// ```

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):

  %named_conv = transform.structured.match ops{["linalg.conv_2d_nchw_fchw"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  

  // Step 1. Tile to forall and sequential scf.for.
  // ======================================================
  %forall_l1, %conv_l1 =
    transform.structured.tile_to_forall_op %named_conv
      num_threads [1]
      ( mapping = [#gpu.block<x>] )
  transform.iree.populate_workgroup_count_region_using_num_threads_slice
    %forall_l1 : (!pdl.operation) -> ()

  // Step 2. Tile to sequential scf.for. 
  // First level with some interchange and second level with sizes if 1 to 
  // properly target decomposition and vectorization.
  // ======================================================
  %conv_l2, %loops_l2:7 = transform.structured.tile_to_scf_for %conv_l1 
  //           N,  F, OH, OW,  C, KH, KW
              [4,  4,  4,  6,  8,  3,  3]
    { interchange = [1, 5, 6, 0, 4, 2, 3] }

  // Decompose needs both OH/KH or OW/KW to be tiled to 1.
  // This is required to further enable vectorization.
  %conv_l3, %loops_l3:7 = transform.structured.tile_to_scf_for %conv_l2 
      //          N,  F, OH, OW,  C, KH, KW
                 [1,  4,  1,  6,  1,  1,  3]

  // Step 3. Force padding of all dimensions and hoist the lhs/rhs pad ops.
  // ======================================================================
  // TODO: hoisting the output requires additional support atm.
  %conv_padded_l3 = transform.structured.pad %conv_l3 {
    padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32], 
    padding_dimensions = [0, 1, 2, 3, 4, 5, 6], 
    pack_paddings=[1, 1, 1]
  }
  transform.iree.apply_patterns %variant_op {canonicalization, cse, licm}
    : (!pdl.operation) -> ()

  %pad_input = transform.get_producer_of_operand %conv_padded_l3[0]
    : (!pdl.operation) -> !transform.op<"tensor.pad">
  transform.structured.hoist_pad %pad_input by 4 loops
     : (!transform.op<"tensor.pad">) -> !pdl.operation

  %pad_kernel = transform.get_producer_of_operand %conv_padded_l3[1]
    : (!pdl.operation) -> !transform.op<"tensor.pad">
  transform.structured.hoist_pad %pad_kernel by 3 loops
     : (!transform.op<"tensor.pad">) -> !pdl.operation

  // Step 4. Decompose to 1-D conv to enable vectorization.
  // ======================================================
  // Decompose needs both OH/KH or OW/KW to be tiled to 1.
  // This is required to further enable vectorization.
  %0 = transform.structured.decompose %conv_padded_l3
  transform.iree.apply_patterns %variant_op { canonicalize, cse }
    : (!pdl.operation) -> ()
  

  // Step 5. Vectorize.
  // ======================================================
  %func_v = transform.structured.match ops{["func.func"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func_v { rank_reducing_linalg }
    : (!pdl.operation) -> ()
  %func_v_2 = transform.structured.vectorize %func_v {vectorize_padding}

  // Step 6. IREE-specific bufferization.
  // ============================================================================
  // Pre-buferization canonicalizations and cleanups help avoid extra copies.
  transform.iree.apply_patterns %variant_op {canonicalization, cse, licm}
    : (!pdl.operation) -> ()
  %variant_op_2 = transform.iree.bufferize %variant_op
    : (!pdl.operation) -> !pdl.operation
  // This is to debug bufferization if needed.
  // %variant_op_2 = transform.iree.bufferize {test_analysis_only, print_conflicts} %variant_op

  // Step 7. IREE-specific cleanup and connection to the runtime and threadpool,
  // required to run e2e.
  // ============================================================================
  %func_e = transform.structured.match ops{["func.func"]} in %variant_op_2
    : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %func_e
    : (!pdl.operation) -> ()
  transform.iree.forall_to_workgroup %func_e
    : (!pdl.operation) -> ()
  
  // Step 8. Late blanket/intrusive lowering of vector ops to vector abstractions
  // that are closer to the LLVM level-of abstraction.
  // ============================================================================
  %func_e_2 = transform.vector.transfer_to_scf %func_e
    max_transfer_rank = 1 full_unroll = true
      : (!pdl.operation) -> !pdl.operation
      
  %func_e_3 = transform.vector.lower_contraction %func_e_2
    lowering_strategy = "outerproduct"
      : (!pdl.operation) -> !pdl.operation

  %func_e_4 = transform.vector.lower_transfer %func_e_3
    max_transfer_rank = 1
      : (!pdl.operation) -> !pdl.operation

  %func_e_5 = transform.vector.lower_transpose %func_e_4
    lowering_strategy = "shuffle"
      : (!pdl.operation) -> !pdl.operation

  %func_e_6 = transform.vector.lower_shape_cast %func_e_5
    : (!pdl.operation) -> !pdl.operation

  // Late canonicalizations and cleanups.
  transform.iree.apply_patterns %variant_op_2 {canonicalization, cse, licm}
    : (!pdl.operation) -> ()
}
