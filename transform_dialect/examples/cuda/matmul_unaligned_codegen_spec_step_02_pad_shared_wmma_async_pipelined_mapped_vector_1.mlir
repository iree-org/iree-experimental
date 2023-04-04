// Instructions; TL;DR
// ===================
//
// Note: this is currently dependent on WIP in the branch:
//   https://github.com/nicolasvasilache/iree/tree/matmul-unaligned
//
// This is an example that should pass but atm fails with:
// ```
// Invalid __global__ read of size 4
// ```
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
//   sed "s/\${M}/999/g" | sed "s/\${K}/1999/g" | sed "s/\${N}/3999/g" | \
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
//      --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_unaligned_codegen_spec_step_02_pad_shared_wmma_async_pipelined_mapped_vector_1.mlir \
//      --iree-codegen-llvmgpu-enable-transform-dialect-jit=false
// ```
//
// To produce PTX:
// ```
//   export IREE_DIR=${HOME}/github/iree; 
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; 
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir | \
//   sed "s/\${M}/999/g" | sed "s/\${K}/1999/g" | sed "s/\${N}/3999/g" | \
//   sed "s/private @matmul_static(/@matmul_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   ${IREE_DIR}/build/tools/iree-compile - \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_unaligned_codegen_spec_step_02_pad_shared_wmma_async_pipelined_mapped_vector_1.mlir \
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
//   sed "s/\${M}/999/g" | sed "s/\${K}/1999/g" | sed "s/\${N}/3999/g" | \
//   sed "s/private @matmul_static(/@matmul_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   ${IREE_DIR}/build/tools/iree-compile - \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_unaligned_codegen_spec_step_02_pad_shared_wmma_async_pipelined_mapped_vector_1.mlir \
//     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
//     --iree-hal-benchmark-dispatch-repeat-count=5 \
//     -o /tmp/foo.vmfb; \
//   scp /tmp/foo.vmfb ${USER}@${A100_MACHINE_IP}:~/ > /dev/null; \
//   ssh ${USER}@${A100_MACHINE_IP} "/usr/local/cuda/bin/nsys profile --stats=true ~/iree-run-module --function=matmul_static --device=cuda --module=foo.vmfb --input=999x1999xf32=1 --input=1999x3999xf32=1 --input=999x3999xf32=1 2>&1" | \
//   grep matmul_static_dispatch | awk '{print $6}'
//
//   # The above prints the min across the 5 invocations.
//   # Alternatively, grep a little more to see what happens in more detail.
//   grep -3 matmul_static_dispatch
// ```
//
// The above command simply prints `370944` (i.e. 0.371 million nanoseconds).
//
//
// Alternatively, run with the profiler:
// ```
//   export IREE_DIR=${HOME}/github/iree; 
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; 
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir | \
//   sed "s/\${M}/999/g" | sed "s/\${K}/1999/g" | sed "s/\${N}/3999/g" | \
//   sed "s/private @matmul_static(/@matmul_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   ${IREE_DIR}/build/tools/iree-compile - \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_unaligned_codegen_spec_step_02_pad_shared_wmma_async_pipelined_mapped_vector_1.mlir \
//     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
//     -o /tmp/foo.vmfb; \
//   scp /tmp/foo.vmfb ${USER}@${A100_MACHINE_IP}:~/ > /dev/null; \
//   ssh ${USER}@${A100_MACHINE_IP} "sudo /usr/local/cuda/bin/ncu -f --set full -o profile ~/iree-run-module --function=matmul_static --device=cuda --module=foo.vmfb \
//     --input=999x1999xf32=1 --input=1999x3999xf32=1 --input=999x3999xf32=1"
// ```
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
    transform.iree.tile_to_forall_and_workgroup_count_region %matmul tile_sizes [128, 128]
      ( mapping = [#gpu.block<y>, #gpu.block<x>] )
  // %fill_l1 = transform.structured.fuse_into_containing_op %fill into %forall_l1
  %matmul_l2, %loops:1 = transform.structured.tile_to_scf_for %matmul_l1 [0, 0, 16]
  // Post-tiling canonicalizations and cleanups.
  transform.iree.apply_patterns %variant_op 
    {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()

  // Step 2. Pad the matmul and force packing to create the buffer in shared memory
  // Note: hoisting here may be dangerous memory-consumption-wise and we may be
  // better off with pipelining only.
  // ==============================================================================
  %matmul_padded_l2 = transform.structured.pad %matmul_l2 {
    padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32], 
    padding_dimensions = [0, 1, 2], 
    pack_paddings=[1, 1, 1]
  }
  // %fill_padded_l1 = transform.structured.pad %fill_l1 {
  //   padding_values = [0.0 : f32, 0.0 : f32], 
  //   padding_dimensions = [0, 1, 2], 
  //   pack_paddings=[0, 1]
  // }

  // Immediately hoist the pad of the output operand to avoid an unpleasant 
  // situation later.
  %pad_res = transform.get_producer_of_operand %matmul_padded_l2[2] 
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
  %insert_lhs = transform.get_producer_of_operand %matmul_padded_l2[0] 
     : (!pdl.operation) -> !pdl.operation
  %copy_lhs = transform.get_producer_of_operand %insert_lhs[0] 
     : (!pdl.operation) -> !pdl.operation
  %extract_lhs = transform.get_producer_of_operand %copy_lhs[1] 
     : (!pdl.operation) -> !pdl.operation
  %fill_lhs = transform.get_producer_of_operand %extract_lhs[0] 
     : (!pdl.operation) -> !pdl.operation
  transform.structured.tile_to_forall_op %fill_lhs num_threads [8, 16]
      ( mapping = [#gpu.linear<y>, #gpu.linear<x>] )
  %forall_copy_lhs, %tiled_copy_lhs = 
    transform.structured.tile_to_forall_op %copy_lhs num_threads [8, 16]
      ( mapping = [#gpu.linear<y>, #gpu.linear<x>] )
  %tiled_copy_lhs_generic = transform.structured.generalize %tiled_copy_lhs
  transform.structured.masked_vectorize %tiled_copy_lhs_generic vector_sizes [16, 1]

  //
  // From now on we can't apply canonicalizations before we lower the masks away.
  //

  // Play catch me if you can with producing fill and copy ...
  // Need to track better through pad rewrite to DPS.
  %insert_rhs = transform.get_producer_of_operand %matmul_padded_l2[1] 
     : (!pdl.operation) -> !pdl.operation
  %copy_rhs = transform.get_producer_of_operand %insert_rhs[0] 
     : (!pdl.operation) -> !pdl.operation
  %extract_rhs = transform.get_producer_of_operand %copy_rhs[1] 
     : (!pdl.operation) -> !pdl.operation
  %fill_rhs = transform.get_producer_of_operand %extract_rhs[0] 
     : (!pdl.operation) -> !pdl.operation
  transform.structured.tile_to_forall_op %fill_rhs num_threads [1, 128]
      ( mapping = [#gpu.linear<y>, #gpu.linear<x>] )
  %forall_copy_rhs, %tiled_copy_rhs = 
    transform.structured.tile_to_forall_op %copy_rhs num_threads [1, 128]
      ( mapping = [#gpu.linear<y>, #gpu.linear<x>] )
  %tiled_copy_rhs_generic = transform.structured.generalize %tiled_copy_rhs
  transform.structured.masked_vectorize %tiled_copy_rhs_generic vector_sizes [16, 1]

  //
  // From now on we can't apply canonicalizations before we lower the masks away.
  //

  // Play catch me if you can with producing fill and copy ... This one is even 
  // nastier because it goes through bbargs -> just rematch.
  // Need to track better through pad rewrite to DPS.
  %copy_res = transform.structured.match ops{["linalg.copy"]} in %variant_op 
    : (!pdl.operation) -> !pdl.operation
  %extract_res = transform.get_producer_of_operand %copy_res[1] 
     : (!pdl.operation) -> !pdl.operation
  %fill_res = transform.get_producer_of_operand %extract_res[0] 
     : (!pdl.operation) -> !pdl.operation
  transform.structured.tile_to_forall_op %fill_res num_threads [1, 128]
      ( mapping = [#gpu.linear<y>, #gpu.linear<x>] )
  %forall_copy_res, %tiled_copy_res = 
    transform.structured.tile_to_forall_op %copy_res num_threads [1, 128]
      ( mapping = [#gpu.linear<y>, #gpu.linear<x>] )
  %tiled_copy_res_generic = transform.structured.generalize %tiled_copy_res
  transform.structured.masked_vectorize %tiled_copy_res_generic vector_sizes [128, 1]

  //
  // From now on we can't apply canonicalizations before we lower the masks away.
  //

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
  %func_v = transform.structured.match ops{["func.func"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func_v { rank_reducing_linalg, rank_reducing_vector }
    : (!pdl.operation) -> ()

  // Lower the masks and we can finally canonicalize / cse.
  %func_v_2 = transform.vector.lower_mask %func_v
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %variant_op 
    {canonicalization, cse, licm} : (!pdl.operation) -> ()

  %func_v_3 = transform.structured.vectorize %func_v_2 { vectorize_padding }

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

  // In the unaligned case, bufferization introduces a copy-back.
  // Due to early hoisting this is annoyingly unmapped and on bufers, need to 
  // tile to scf.forall (and extend it upstream to work with buffers).
  %mapped_copy_back = 
    transform.structured.match ops{["linalg.generic"]} in %variant_op_3
      : (!pdl.operation) -> !pdl.operation
  %forall_mapped_copy_back, %tiled_mapped_copy_back = 
    transform.structured.tile_to_forall_op %mapped_copy_back num_threads [1, 128]
      ( mapping = [#gpu.linear<y>, #gpu.linear<x>] )
  transform.structured.masked_vectorize %tiled_mapped_copy_back vector_sizes [128, 1]
  // Lower vector + canonicalize / licm again.
  %func_m_x = transform.vector.lower_mask %func_m
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %variant_op_3
    {canonicalization, cse, licm} : (!pdl.operation) -> ()

  // Step 8. Post-bufferization mapping blocks/workgroup and threads/subgroup.
  // =========================================================================
  transform.iree.apply_patterns %variant_op_3 
    {canonicalization, cse, licm, tiling_canonicalization}
    : (!pdl.operation) -> ()
  transform.iree.forall_to_workgroup %func_m_x : (!pdl.operation) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_m_x
      workgroup_dims = [64, 2, 1] warp_dims = [2, 2, 1]
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
  transform.iree.apply_patterns %func_m_x {canonicalization, cse, licm}
    : (!pdl.operation) -> ()
  transform.iree.hoist_static_alloc %func_m_x : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func_m_x { fold_memref_aliases }
    : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func_m_x { extract_address_computations }
    : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func_m_x {canonicalization, cse, licm}
    : (!pdl.operation) -> ()
  // TODO: This currently crashes without Thomas' hack.
  transform.iree.apply_patterns %func_m_x { unroll_vectors_gpu_wmma }
    : (!pdl.operation) -> ()

  // Hoist redundant vector transfers to allow vectorization to proceed.
  // We really don't want to do this after bufferization but we need to atm.
  // One way to work around this is to hoist the pad ops on the output earlier 
  // but this has other tradeoffs. Still needs investigation.
  %func_m_8 = transform.structured.hoist_redundant_vector_transfers %func_m_x
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

  //===---------------------------------------------------------------------===//
  // DSmeom out of bounds memory access here. This needs investigation.
  // ========= Invalid __global__ read of size 4
  // =========     at 0x00001f30 in matmul_static_dispatch_0_matmul_1000x4000x2000
  // =========     by thread (63,1,0) in block (31,2,0)
  // =========     Address 0x7f8153e84800 is out of bounds
  // =========     Saved host backtrace up to driver entry point at kernel launch time
  // =========     Host Frame:/lib/x86_64-linux-gnu/libcuda.so [0x2e9441]
  //===---------------------------------------------------------------------===//

  // // Step 9. Multi-buffering.
  // // =========================================================================
  // transform.iree.apply_patterns %func_m_10 {canonicalize, cse}
  //   : (!pdl.operation) -> ()
  // // Fold memref aliases to allow multi-buffering to proceed.
  // transform.iree.apply_patterns %func_m_10 { fold_memref_aliases }
  //   : (!pdl.operation) -> ()
  // %allocs = transform.structured.match ops{["memref.alloc"]} in %func_m_10
  //   : (!pdl.operation) -> !transform.op<"memref.alloc">
  // %mb_allocs = transform.memref.multibuffer %allocs {factor = 3 : i64, skip_analysis } 
  //   : (!transform.op<"memref.alloc">) -> !pdl.operation

  // // TODO: cp-async currently creates errors such as:
  // // Misaligned Shared or Local Address
  // // =========     at 0x00001120 in matmul_static_dispatch_0_matmul_999x3999x1999 
  
  // // Step 10. Cp-async.
  // // ===========================================================================
  // // Lower remaining vector ops to 1-D which will trigger the cp-async.
  // // Alternatively we could explicitly unroll to 1-D innermost vectors if we 
  // // wanted a specific target shape.
  // transform.iree.create_async_groups %func_m_10 {use_mma_sync = false} 
  //   : (!pdl.operation) -> ()
  // transform.iree.apply_patterns %func_m_10 {canonicalize, cse, fold_memref_aliases, licm}
  //   : (!pdl.operation) -> ()

  // // Step 11. Pipeline shared memory copies.
  // // ===========================================================================
  // %mma_compute = transform.structured.match ops{["gpu.subgroup_mma_compute"]} in %variant_op_3
  //   : (!pdl.operation) -> !pdl.operation
  // // Pre pipelining cleanups.
  // transform.iree.apply_patterns %func_m_10 {canonicalization, cse}
  //   : (!pdl.operation) -> ()
  // %for = transform.loop.get_parent_for %mma_compute : (!pdl.operation) -> !transform.op<"scf.for">
  // %pipelined_for = transform.iree.pipeline_shared_memory_copies %for { depth = 3 } 
  //   : (!transform.op<"scf.for">) -> !transform.op<"scf.for">

  // Late canonicalizations and cleanups.
  transform.iree.apply_patterns %variant_op_3
    {canonicalization, cse, licm, tiling_canonicalization}
    : (!pdl.operation) -> ()
}
