// Instructions; TL;DR
// ===================
//
// This is an attempt to reproduce a pipelining related issue that results in
// illegal memory accesses on matmul examples.
// The point of doing it on a much smaller reduction kernel is that the IR is 
// much smaller and easier to grok.
//
// Atm the pipelining does not kick in, as the pattern matching part seems quite
// fragile.
//
// As a nice byproduct, this results in a pipelined reduction kernel that seems 
// quite more efficient than the warp-shuffle based kernels in certain cases, 
// even though the compute is extremely wasteful and the mapping is extremely 
// naive atm.
//
// We should investigate deeper the usage of cp.async + pipelining as a way to 
// generalized performance on reductions.
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/reduction.mlir |\
//   sed "s/\${SZ1}/1/g" | sed "s/\${SZ2}/1/g" | sed "s/\${SZ3}/1/g" | \
//   sed "s/\${ELEMENTAL_TYPE}/f32/g" | sed "s/\${ZERO}/0.0/g" | sed "s/\${ADD_OP}/arith.addf/g" | sed "s/\${DIV_OP}/arith.divf/g" | \
//   sed "s/private @reduction_no_fill_2d_dynamic(/@reduction_no_fill_2d_dynamic(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   ${IREE_DIR}/build/tools/iree-opt \
//     --iree-hal-target-backends=cuda \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_DIR}/build/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
//      --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/reduction_pipelined_spec.mlir \
//      --iree-codegen-llvmgpu-enable-transform-dialect-jit=false
/// ```
//
// To run e2e on a remote machine (${A100_MACHINE_IP}) with an A100 GPU:
// ```
//   # Do this only once:
//   # scp ${IREE_DIR}/build/tools/iree-run-module ${USER}@${A100_MACHINE_IP}:~/;
//
//   export IREE_DIR=${HOME}/github/iree; 
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; 
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/reduction.mlir |\
//   sed "s/\${SZ1}/1/g" | sed "s/\${SZ2}/1/g" | sed "s/\${SZ3}/1/g" | \
//   sed "s/\${ELEMENTAL_TYPE}/f32/g" | sed "s/\${ZERO}/0.0/g" | sed "s/\${ADD_OP}/arith.addf/g" | sed "s/\${DIV_OP}/arith.divf/g" | \
//   sed "s/private @reduction_no_fill_2d_dynamic(/@reduction_no_fill_2d_dynamic(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   ${IREE_DIR}/build/tools/iree-compile - \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/reduction_pipelined_spec.mlir \
//     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
//     --iree-hal-benchmark-dispatch-repeat-count=5 \
//     -o /tmp/foo.vmfb; \
//   scp /tmp/foo.vmfb ${USER}@${A100_MACHINE_IP}:~/ > /dev/null; \
//   ssh ${USER}@${A100_MACHINE_IP} "/usr/local/cuda/bin/nsys profile --stats=true ~/iree-run-module --function=reduction_no_fill_2d_static --device=cuda --module=foo.vmfb --input=123x2511xf32=1 --input=123xf32=1 2>&1"
// ```
//

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %generic = transform.structured.match ops{["linalg.generic"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  %forall_l1, %generic_l1 =
    transform.iree.tile_to_forall_and_workgroup_count_region %generic tile_sizes [3]
      ( mapping = [#gpu.block<x>] )
  %generic_l2, %loops:1 = transform.structured.tile_to_scf_for %generic_l1 [0, 32]

  // Step 2. Pad, force packing abd hoist to create the buffer in shared memory.
  // ==============================================================================
  %generic_padded_l2 = transform.structured.pad %generic_l2 {
    padding_values = [0.0 : f32, 0.0 : f32], 
    padding_dimensions = [0, 1], 
    pack_paddings=[1, 1]
  }

  %pad_res = transform.get_producer_of_operand %generic_padded_l2[1] 
     : (!pdl.operation) -> !pdl.operation
  %pad_res_2 = transform.cast %pad_res : !pdl.operation to !transform.op<"tensor.pad">
  transform.structured.hoist_pad %pad_res_2 by 1 loops
     : (!transform.op<"tensor.pad">) -> !pdl.operation

  // Step 3. Rewrite tensor.pad in DPS, this creates linalg.copy ops.
  // ================================================================
  %pad = transform.structured.match ops{["tensor.pad"]} in %variant_op 
    : (!pdl.operation) -> !pdl.operation
  %padded = transform.structured.rewrite_in_destination_passing_style %pad 
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %variant_op 
    {canonicalization} : (!pdl.operation) -> ()

  // Step 4. Map to threads, **SIMT** programming model.
  // ===================================================
  // Play catch me if you can with producing fill and copy ...
  // Need to track better through pad rewrite to DPS.
  %insert_lhs = transform.get_producer_of_operand %generic_padded_l2[0]
     : (!pdl.operation) -> !pdl.operation
  %copy_lhs = transform.get_producer_of_operand %insert_lhs[0] 
     : (!pdl.operation) -> !pdl.operation
  %extract_lhs = transform.get_producer_of_operand %copy_lhs[1] 
     : (!pdl.operation) -> !pdl.operation
  %fill_lhs = transform.get_producer_of_operand %extract_lhs[0] 
     : (!pdl.operation) -> !pdl.operation
  transform.structured.tile_to_forall_op %fill_lhs num_threads [1, 32]
      ( mapping = [#gpu.linear<y>, #gpu.linear<x>] )
  %forall_copy_lhs, %tiled_copy_lhs = 
    transform.structured.tile_to_forall_op %copy_lhs num_threads [1, 32]
      ( mapping = [#gpu.linear<y>, #gpu.linear<x>] )
  %tiled_copy_lhs_generic = transform.structured.generalize %tiled_copy_lhs
  transform.structured.masked_vectorize %tiled_copy_lhs_generic vector_sizes [3, 1]

  // Step 3. Vectorize.
  // ============================================================================
  %func_v = transform.structured.match ops{["func.func"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  // Lower the masks and we can finally canonicalize / cse.
  %func_v_2 = transform.vector.lower_mask %func_v
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %variant_op 
    {canonicalization, cse, licm} : (!pdl.operation) -> ()
    
  transform.iree.apply_patterns %func_v_2 { rank_reducing_linalg, rank_reducing_vector }
    : (!pdl.operation) -> ()
  %func_v_3 = transform.structured.vectorize %func_v_2 { vectorize_padding }

  // // Step 4. Bufferize.
  // // ============================================================================
  // // TODO: eliminate tensor.empty ops and make sure we bufferize inplace.
  // %empty_tensor = transform.structured.match ops{["tensor.empty"]} in %module_op
  //   : (!pdl.operation) -> !transform.op<"tensor.empty">
  // transform.bufferization.empty_tensor_to_alloc_tensor %empty_tensor
  //   : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
  // transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %module_op 
  //   {bufferize_function_boundaries = true, allow_return_allocs = true}
  // %func_3 = transform.structured.vectorize %func_2

  // Step 7. Bufferize and drop HAL descriptor from memref ops.
  // ==========================================================
  // Pre-bufferization canonicalizations and cleanups help avoid extra copies.
  // transform.iree.apply_patterns %variant_op {canonicalization, cse, licm}
  //   : (!pdl.operation) -> ()
  transform.iree.eliminate_empty_tensors %variant_op : (!pdl.operation) -> ()
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op
    : (!pdl.operation) -> (!pdl.operation)
  %func_m = transform.structured.match ops{["func.func"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %func_m
    : (!pdl.operation) -> ()
  transform.iree.apply_buffer_optimizations %func_m : (!pdl.operation) -> ()

  // Step 8. Post-bufferization mapping blocks/workgroup and threads/subgroup.
  // =========================================================================
  transform.iree.apply_patterns %variant_op_3 
    {canonicalization, cse, licm, tiling_canonicalization}
    : (!pdl.operation) -> ()
  transform.iree.forall_to_workgroup %func_m : (!pdl.operation) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_m
      workgroup_dims = [32, 1, 1]
    : (!pdl.operation) -> ()


  //===---------------------------------------------------------------------===//
  // BEGIN - Annoying phase-ordered section
  //===---------------------------------------------------------------------===//
  // Vector transfer_read and transfer_write patterns have different subview
  // folding behavior, force a fold_memref_aliases on them to enable redundant
  // vector transfer hoisting.
  // Unfortunately, fold_memref_aliases breaks vector_to_mma conversion across 
  // scf.for after unrolling dur to insert_strided_slice / extract_strided_slice
  // across iter_args boundaries.
  transform.iree.apply_patterns %func_m {canonicalization, cse, licm}
    : (!pdl.operation) -> ()
  transform.iree.hoist_static_alloc %func_m : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func_m { fold_memref_aliases }
    : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func_m { extract_address_computations }
    : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func_m {canonicalization, cse, licm}
    : (!pdl.operation) -> ()
  // TODO: This currently crashes without Thomas' hack.
  transform.iree.apply_patterns %func_m { unroll_vectors_gpu_wmma }
    : (!pdl.operation) -> ()

  // Hoist redundant vector transfers to allow vectorization to proceed.
  // We really don't want to do this after bufferization but we need to atm.
  // One way to work around this is to hoist the pad ops on the output earlier 
  // but this has other tradeoffs. Still needs investigation.
  %func_m_8 = transform.structured.hoist_redundant_vector_transfers %func_m
    : (!pdl.operation) -> !pdl.operation

  transform.iree.apply_patterns %func_m_8 {canonicalization, cse, licm}
    : (!pdl.operation) -> ()
  transform.iree.apply_buffer_optimizations %func_m_8 : (!pdl.operation) -> ()

  // This must occur after bufferization, unrolling and hoisting because of the
  // fancy CUDA types.
  transform.iree.vector.vector_to_mma_conversion %func_m_8 { use_wmma }
    : (!pdl.operation) -> ()
  //===---------------------------------------------------------------------===//
  // END - Annoying phase-ordered section
  //===---------------------------------------------------------------------===//
  // Lower vector.transfer to 1-D early so we can optionally avoid steps 9-11 
  // and ensure proper execution in a simpler case.
  // Lowering to 1-D vector is necessary anyway for cp.async to be generated.
  %func_m_9 = transform.vector.transfer_to_scf %func_m_8
    max_transfer_rank = 1 full_unroll = true
      : (!pdl.operation) -> !pdl.operation
  %func_m_10 = transform.vector.lower_mask %func_m_9
      : (!pdl.operation) -> !pdl.operation


  // Step 9. Multi-buffering.
  // =========================================================================
  transform.iree.apply_patterns %func_m_10 {canonicalize, cse}
    : (!pdl.operation) -> ()
  // Fold memref aliases to allow multi-buffering to proceed.
  transform.iree.apply_patterns %func_m_10 { fold_memref_aliases }
    : (!pdl.operation) -> ()
  %allocs = transform.structured.match ops{["memref.alloc"]} in %func_m_10
    : (!pdl.operation) -> !transform.op<"memref.alloc">
  %mb_allocs = transform.memref.multibuffer %allocs {factor = 5 : i64, skip_analysis } 
    : (!transform.op<"memref.alloc">) -> !pdl.operation

  // Step 10. Cp-async.
  // ===========================================================================
  // Lower remaining vector ops to 1-D which will trigger the cp-async.
  // Alternatively we could explicitly unroll to 1-D innermost vectors if we 
  // wanted a specific target shape.
  transform.iree.create_async_groups %func_m_10 {use_mma_sync = false} 
    : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func_m_10 {canonicalize, cse, fold_memref_aliases, licm}
    : (!pdl.operation) -> ()


  // Step 11. Pipeline shared memory copies.
  // ===========================================================================
  %mma_compute = transform.structured.match ops{["gpu.subgroup_mma_compute"]} in %variant_op_3
    : (!pdl.operation) -> !pdl.operation
  // Pre pipelining cleanups.
  transform.iree.apply_patterns %func_m_10 {canonicalization, cse}
    : (!pdl.operation) -> ()
  %for = transform.loop.get_parent_for %mma_compute : (!pdl.operation) -> !transform.op<"scf.for">
  %pipelined_for = transform.iree.pipeline_shared_memory_copies %for { depth = 5 } 
    : (!transform.op<"scf.for">) -> !transform.op<"scf.for">

  // Late canonicalizations and cleanups.
  transform.iree.apply_patterns %variant_op_3
    {canonicalization, cse, licm, tiling_canonicalization}
    : (!pdl.operation) -> ()
}
