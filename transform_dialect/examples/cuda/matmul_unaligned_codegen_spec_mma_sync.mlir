// Instructions; TL;DR
// ===================
//
// Note: this is currently dependent on IREE + LLVM WIP PRs that have not yet 
// landed.
//
// Note: this is currently using vector 4 so it only works if all tensors have
// sizes divisible by 4 in the f32 case.
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
//   sed "s/\${M}/3452/g" | sed "s/\${N}/1020/g" | sed "s/\${K}/2044/g" | \
//   sed "s/private @matmul_static(/@matmul_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   ${IREE_DIR}/build/tools/iree-opt \
//     --iree-hal-target-backends=cuda \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_DIR}/build/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
//      --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_unaligned_codegen_spec_mma_sync.mlir \
//      --iree-codegen-llvmgpu-enable-transform-dialect-jit=false
// ```
//
// To produce PTX:
// ```
//   export IREE_DIR=${HOME}/github/iree; 
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; 
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir | \
//   sed "s/\${M}/3452/g" | sed "s/\${N}/1020/g" | sed "s/\${K}/2044/g" | \
//   sed "s/private @matmul_static(/@matmul_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   ${IREE_DIR}/build/tools/iree-compile - \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_unaligned_codegen_spec_mma_sync.mlir \
//     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false 
// ```
//
// To run e2e on a remote machine (${A100_MACHINE_IP}) with an A100 GPU:
// ```
//   # Do this only once:
//   # scp ${IREE_DIR}/build/tools/iree-run-module ${USER}@${A100_MACHINE_IP}:~/;
//
//   export IREE_DIR=${HOME}/github/iree; 
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; 
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir | \
//   sed "s/\${M}/3452/g" | sed "s/\${N}/1020/g" | sed "s/\${K}/2044/g" | \
//   sed "s/private @matmul_static(/@matmul_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   ${IREE_DIR}/build/tools/iree-compile - \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_unaligned_codegen_spec_mma_sync.mlir \
//     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
//     --iree-hal-benchmark-dispatch-repeat-count=5 \
//     -o /tmp/foo.vmfb; \
//   scp /tmp/foo.vmfb ${USER}@${A100_MACHINE_IP}:~/ > /dev/null; \
//   ssh ${USER}@${A100_MACHINE_IP} "/usr/local/cuda/bin/nsys profile --stats=true ~/iree-run-module --function=matmul_static --device=cuda --module=foo.vmfb --input=3452x2044xf32=1 --input=2044x1020xf32=1 --input=3452x1020xf32=1 2>&1"
// ```
//
// Alternatively, run with the profiler: 
//   `sudo /usr/local/cuda/bin/ncu -f --set full -o profile ~/iree-run-module ...`
//
transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  // No fill for now, add it when we can do matmul.
  //
  // %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
  //   : (!pdl.operation) -> !pdl.operation
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
 
  // Step 1. Tile to forall and sequential scf.for.
  // ======================================================
  %forall_l1, %matmul_l1 =
    transform.structured.tile_to_forall_op %matmul tile_sizes [128, 128]
      ( mapping = [#gpu.block<y>, #gpu.block<x>] )
  transform.iree.populate_workgroup_count_region_using_num_threads_slice
    %forall_l1 : (!pdl.operation) -> ()
  // %fill_l1 = transform.structured.fuse_into_containing_op %fill into %forall_l1
  %matmul_l2, %loops:1 = transform.structured.tile_to_scf_for %matmul_l1 [0, 0, 16]
  // Post-tiling canonicalizations and cleanups.
  transform.iree.apply_patterns %variant_op 
    {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
 
  // Step 2. Pad the matmul and force packing to create the buffer in shared memory
  // ==============================================================================
  %matmul_padded_l2 = transform.structured.pad %matmul_l2 {
    padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32], 
    padding_dimensions = [0, 1, 2], 
    pack_paddings=[0, 0, 0]
  }
 
  // Immediately hoist the pad of the output operand to avoid an unpleasant 
  // situation later.
  %pad_res = transform.get_producer_of_operand %matmul_padded_l2[2] 
     : (!pdl.operation) -> !pdl.operation
  %pad_res_2 = transform.cast %pad_res : !pdl.operation to !transform.op<"tensor.pad">
  %pad_res_hoisted = transform.structured.hoist_pad %pad_res_2 by 1 loops
     : (!transform.op<"tensor.pad">) -> !pdl.operation
  %insert_slice_back = transform.structured.match ops{["tensor.insert_slice"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  %copy_back = transform.structured.insert_slice_to_copy %insert_slice_back
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %variant_op 
    {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()
 

  // Step 4. Map to threads, **SIMT** programming model.
  // ===================================================
  %pad_lhs = transform.get_producer_of_operand %matmul_padded_l2[0] 
     : (!pdl.operation) -> !pdl.operation
  %forall_pad_lhs, %tiled_pad_lhs = 
    transform.structured.tile_to_forall_op %pad_lhs num_threads [32, 4]
      ( mapping = [#gpu.linear<y>, #gpu.linear<x>] )
  %if_lhs = transform.structured.match ops{["scf.if"]} in %forall_pad_lhs 
    : (!pdl.operation) -> !transform.any_op
  transform.scf.take_assumed_branch %if_lhs take_else_branch 
    : (!transform.any_op) -> ()

  %pad_rhs = transform.get_producer_of_operand %matmul_padded_l2[1] 
     : (!pdl.operation) -> !pdl.operation
  %forall_pad_rhs, %tiled_pad_rhs = 
    transform.structured.tile_to_forall_op %pad_rhs num_threads [4, 32]
      ( mapping = [#gpu.linear<y>, #gpu.linear<x>] )
  %if_rhs = transform.structured.match ops{["scf.if"]} in %forall_pad_rhs 
    : (!pdl.operation) -> !transform.any_op
  transform.scf.take_assumed_branch %if_rhs take_else_branch 
    : (!transform.any_op) -> ()

  %forall_pad_res, %tiled_pad_res = 
    transform.structured.tile_to_forall_op %pad_res_hoisted num_threads [4, 32]
      ( mapping = [#gpu.linear<y>, #gpu.linear<x>] )
  transform.iree.apply_patterns %variant_op 
    {canonicalization, cse, licm} : (!pdl.operation) -> ()
  %if_res = transform.structured.match ops{["scf.if"]} in %forall_pad_res 
    : (!pdl.operation) -> !transform.any_op
  transform.scf.take_assumed_branch %if_res take_else_branch 
    : (!transform.any_op) -> ()

  %forall_copy_back, %tiled_copy_back = 
    transform.structured.tile_to_forall_op %copy_back num_threads [4, 32]
      ( mapping = [#gpu.linear<y>, #gpu.linear<x>] )
  transform.iree.apply_patterns %variant_op 
    {canonicalization, cse, licm} : (!pdl.operation) -> ()

  // Masked vectorize prevents canonicalizations to occur until they are lowered.
  // This is because they only allow a single masked op and canonicalization will
  // pull ops inside the mask region, emitting invalid IR.
  transform.structured.masked_vectorize %tiled_pad_lhs vector_sizes [4, 4]
  transform.structured.masked_vectorize %tiled_pad_rhs vector_sizes [4, 4]
  transform.structured.masked_vectorize %tiled_pad_res vector_sizes [32, 4]
  transform.structured.masked_vectorize %tiled_copy_back vector_sizes [32, 4]

  // Step 5. Contraction part mapped to threads with a **SIMD** programming model.
  // =============================================================================
  %forall_l3, %matmul_padded_l3 = 
    transform.structured.tile_to_forall_op %matmul_padded_l2 num_threads [2, 2]
      ( mapping = [#gpu.warp<y>, #gpu.warp<x>] )
  // %forall_fill_l3, %fill_l3 = 
  //   transform.structured.tile_to_forall_op %fill_padded_l1 num_threads [2, 2]
  //     ( mapping = [#gpu.warp<y>, #gpu.warp<x>] )
 
  // Step 6. Rank-reduce and vectorize.
  // ==================================
  // Lower the masks to allow canonicalizations to kick in.
  %func_v = transform.structured.match ops{["func.func"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  %func_v_2 = transform.vector.lower_masked_transfers %func_v
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func_v_2 { rank_reducing_linalg, rank_reducing_vector }
    : (!pdl.operation) -> () 
  %func_v_3 = transform.structured.vectorize %func_v_2
 
  // Step 7. Bufferize and drop HAL descriptor from memref ops.
  // ==========================================================
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
  transform.iree.apply_buffer_optimizations %func_m : (!pdl.operation) -> ()
 
  // Step 8. Post-bufferization mapping blocks/workgroup and threads/subgroup.
  // =========================================================================
  transform.iree.apply_patterns %variant_op_3 
    {canonicalization, cse, licm, tiling_canonicalization}
    : (!pdl.operation) -> ()
  transform.iree.forall_to_workgroup %func_m : (!pdl.operation) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_m
      workgroup_dims = [64, 2, 1] warp_dims = [2, 2, 1]
    : (!pdl.operation) -> ()
 
  //===---------------------------------------------------------------------===//
  // BEGIN - Annoying phase-ordered section
  //===---------------------------------------------------------------------===//
  // Vector transfer_read and transfer_write patterns have different subview
  // folding behavior, force a fold_memref_aliases on them to enable redundant
  // vector transfer hoisting.
  // Unfortunately, fold_memref_aliases breaks vector_to_mma conversion across 
  // scf.for after unrolling due to insert_strided_slice / extract_strided_slice
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
  transform.iree.apply_patterns %func_m { unroll_vectors_gpu_mma_sync }
    : (!pdl.operation) -> ()
 
  // Hoist redundant vector transfers to allow vectorization to proceed.
  // We really don't want to do this after bufferization but we need to atm.
  // One way to work around this is to hoist the pad ops on the output earlier 
  // but this has other tradeoffs. Still needs investigation.
  %func_m_2 = transform.structured.hoist_redundant_vector_transfers %func_m
    : (!pdl.operation) -> !pdl.operation
 
  transform.iree.apply_patterns %func_m_2 {canonicalization, cse, licm}
    : (!pdl.operation) -> ()
  transform.iree.apply_buffer_optimizations %func_m_2 : (!pdl.operation) -> ()
 
  // This must occur after bufferization, unrolling and hoisting because of the
  // fancy CUDA types.
  transform.iree.vector.vector_to_mma_conversion %func_m_2 { use_mma_sync }
    : (!pdl.operation) -> ()
  //===---------------------------------------------------------------------===//
  // END - Annoying phase-ordered section
  //===---------------------------------------------------------------------===//
 
  // Lower vector.transfer to 1-D early so we can optionally avoid steps 9-11 
  // and ensure proper execution in a simpler case.
  // Lowering to 1-D vector is necessary anyway for cp.async to be generated.
  %func_m_3 = transform.vector.transfer_to_scf %func_m_2
    max_transfer_rank = 1 full_unroll = true
      : (!pdl.operation) -> !pdl.operation
 
  // Step 9. Multi-buffering.
  // =========================================================================
  %func_mb = transform.structured.match ops{["func.func"]} in %variant_op_3
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func_mb {canonicalize, cse}
    : (!pdl.operation) -> ()
  // Fold memref aliases to allow multi-buffering to proceed.
  transform.iree.apply_patterns %func_mb { fold_memref_aliases }
    : (!pdl.operation) -> ()
  %allocs = transform.structured.match ops{["memref.alloc"]} in %func_mb
    : (!pdl.operation) -> !transform.op<"memref.alloc">
  %mb_allocs = transform.memref.multibuffer %allocs {factor = 3 : i64, skip_analysis } 
    : (!transform.op<"memref.alloc">) -> !pdl.operation
 
  // Step 10. Cp-async.
  // ===========================================================================
  // Lower remaining vector ops to 1-D which will trigger the cp-async.
  // Alternatively we could explicitly unroll to 1-D innermost vectors if we 
  // wanted a specific target shape.
  transform.iree.apply_patterns %variant_op_3
    {canonicalization, cse, licm, tiling_canonicalization}
    : (!pdl.operation) -> ()
  %func_m_cp = transform.structured.match ops{["func.func"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  transform.iree.create_async_groups %func_m_cp {use_mma_sync = true} 
    : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func_m_cp {canonicalize, cse, fold_memref_aliases, licm}
    : (!pdl.operation) -> ()
 
  // Step 11. Pipeline shared memory copies.
  // ===========================================================================
  %mma_compute = transform.structured.match ops{["gpu.subgroup_mma_compute"]} in %variant_op_3
    : (!pdl.operation) -> !pdl.operation
  // Pre pipelining cleanups.
  transform.iree.apply_patterns %func_m_cp {canonicalization, cse}
    : (!pdl.operation) -> ()
  %for = transform.loop.get_parent_for %mma_compute : (!pdl.operation) -> !transform.op<"scf.for">
  %pipelined_for = transform.iree.pipeline_shared_memory_copies %for { depth = 3 } 
    : (!transform.op<"scf.for">) -> !transform.op<"scf.for">

  %func_m_cp_2 = transform.vector.lower_masks %func_m_cp
      : (!pdl.operation) -> !pdl.operation
  %func_m_cp_3 = transform.vector.materialize_masks %func_m_cp_2
      : (!pdl.operation) -> !pdl.operation
 
  // Late canonicalizations and cleanups.
  transform.iree.apply_patterns %variant_op_3
    {canonicalization, cse, licm, tiling_canonicalization}
    : (!pdl.operation) -> ()
}
