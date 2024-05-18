// Instructions; TL;DR
// ===================
//
// This script shows an example of tiling for 2 levels with:
//   - padding with nofold to achieve shared memory mapping
//   - linear thread mapping of the copies to shared memory
//   - connecting to wmma operations.
//   - multi-buffering and async copy pipelining.
//
// This still exhibits some phase-ordering issues that need to be better fixed
// over time.
//
// Steps forward are:
// 
// 0. make things work with tensors to reduce phase ordering issues. This is 
//    still ambiguous.
// 1. propagate insert_strided_slice across loop and create more iter_args
//    after unrolling which would allow insert/extract on vectors to fold away 
//    and reenable wmma slice analysis
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
//   sed "s/\${M}/3456/g" | sed "s/\${K}/1024/g" | sed "s/\${N}/2048/g" | \
//   sed "s/private @fill_matmul_static(/@fill_matmul_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   ${IREE_DIR}/build/tools/iree-opt \
//     --iree-hal-target-backends=cuda \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_DIR}/build/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
//      --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_codegen_spec_step_02_pad_shared_wmma_async_pipelined_mapped.mlir \
//      --iree-codegen-llvmgpu-enable-transform-dialect-jit=false
// ```
//
// To produce PTX:
// ```
//   export IREE_DIR=${HOME}/github/iree; 
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; 
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir | \
//   sed "s/\${M}/3456/g" | sed "s/\${K}/1024/g" | sed "s/\${N}/2048/g" | \
//   sed "s/private @fill_matmul_static(/@fill_matmul_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   ${IREE_DIR}/build/tools/iree-compile - \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_codegen_spec_step_02_pad_shared_wmma_async_pipelined_mapped.mlir \
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
//   sed "s/\${M}/3456/g" | sed "s/\${K}/1024/g" | sed "s/\${N}/2048/g" | \
//   sed "s/private @fill_matmul_static(/@fill_matmul_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   ${IREE_DIR}/build/tools/iree-compile - \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_codegen_spec_step_02_pad_shared_wmma_async_pipelined_mapped.mlir \
//     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
//     --iree-hal-benchmark-dispatch-repeat-count=5 \
//     -o /tmp/foo.vmfb; \
//   scp /tmp/foo.vmfb ${USER}@${A100_MACHINE_IP}:~/ > /dev/null; \
//   ssh ${USER}@${A100_MACHINE_IP} "/usr/local/cuda/bin/nsys profile --stats=true ~/iree-run-module --function=fill_matmul_static --device=cuda --module=foo.vmfb --input=3456x1024xf32=1 --input=1024x2048xf32=1 --input=3456x2048xf32=1 2>&1" | \
//   grep fill_matmul_static_dispatch | awk '{print $6}'
//
//   # The above prints the min across the 5 invocations.
//   # Alternatively, grep a little more to see what happens in more detail.
//   grep -3 fill_matmul_static_dispatch
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
//   sed "s/\${M}/3456/g" | sed "s/\${K}/1024/g" | sed "s/\${N}/2048/g" | \
//   sed "s/private @fill_matmul_static(/@fill_matmul_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   ${IREE_DIR}/build/tools/iree-compile - \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_codegen_spec_step_02_pad_shared_wmma_async_pipelined_mapped.mlir \
//     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
//     -o /tmp/foo.vmfb; \
//   scp /tmp/foo.vmfb ${USER}@${A100_MACHINE_IP}:~/ > /dev/null; \
//   ssh ${USER}@${A100_MACHINE_IP} "sudo /usr/local/cuda/bin/ncu -f --set full -o profile ~/iree-run-module --function=fill_matmul_static --device=cuda --module=foo.vmfb \
//     --input=3456x1024xf32=1 --input=1024x2048xf32=1 --input=3456x2048xf32=1"
// ```
//
//
// CHECK: hal.interface.workgroup.id[1] : index
// CHECK: hal.interface.workgroup.id[0] : index
// CHECK: gpu.thread_id  x
// CHECK: gpu.thread_id  y
//
// TODO: C should either be fused with a consumer or better interleaved and we shouldn't start with a large blocking read to global.
//
// CHECK-COUNT-16: gpu.subgroup_mma_load_matrix %{{.*}} {leadDimension = 2048 : index} : memref<64x64xf32, strided<[2048, 1], offset: ?>> -> !gpu.mma_matrix<16x16xf32, "COp">
//
//  CHECK-COUNT-4:   nvgpu.device_async_copy %{{.*}} : memref<4x4xf32, strided<[1024, 1], offset: ?>> to memref<4x4xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
//          CHECK:   nvgpu.device_async_create_group %{{.*}} {__pipelining_first_stage__}
//          CHECK:   gpu.barrier {__pipelining_first_stage__}
//
//  CHECK-COUNT-4:   nvgpu.device_async_copy %{{.*}} : memref<4x4xf32, strided<[2048, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
//          CHECK:   nvgpu.device_async_create_group %{{.*}} {__pipelining_first_stage__}
//  CHECK-COUNT-4:   nvgpu.device_async_copy %{{.*}} : memref<4x4xf32, strided<[1024, 1], offset: ?>> to memref<4x4xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
//          CHECK:   nvgpu.device_async_create_group %{{.*}} {__pipelining_first_stage__}
//          CHECK:   gpu.barrier {__pipelining_first_stage__}
//
//  CHECK-COUNT-4:   nvgpu.device_async_copy %{{.*}} : memref<4x4xf32, strided<[2048, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
//          CHECK:   nvgpu.device_async_create_group %{{.*}} {__pipelining_first_stage__}
//  CHECK-COUNT-4:   nvgpu.device_async_copy %{{.*}} : memref<4x4xf32, strided<[1024, 1], offset: ?>> to memref<4x4xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
//          CHECK:   nvgpu.device_async_create_group %{{.*}} {__pipelining_first_stage__}
//          CHECK:   gpu.barrier {__pipelining_first_stage__}
//
//  CHECK-COUNT-4:   nvgpu.device_async_copy %{{.*}} : memref<4x4xf32, strided<[2048, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
//          CHECK:   nvgpu.device_async_create_group %{{.*}} {__pipelining_first_stage__}
//  CHECK-COUNT-4:   nvgpu.device_async_copy %{{.*}} : memref<4x4xf32, strided<[1024, 1], offset: ?>> to memref<4x4xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
//          CHECK:   nvgpu.device_async_create_group %{{.*}} {__pipelining_first_stage__}
//          CHECK:   gpu.barrier {__pipelining_first_stage__}
//
//  CHECK-COUNT-4:   nvgpu.device_async_copy %{{.*}} : memref<4x4xf32, strided<[2048, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
//          CHECK:   nvgpu.device_async_create_group %{{.*}} {__pipelining_first_stage__}
//  CHECK-COUNT-4:   nvgpu.device_async_copy %{{.*}} : memref<4x4xf32, strided<[1024, 1], offset: ?>> to memref<4x4xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
//          CHECK:   nvgpu.device_async_create_group %{{.*}} {__pipelining_first_stage__}
//          CHECK:   gpu.barrier {__pipelining_first_stage__}
//
//  CHECK-COUNT-4:   nvgpu.device_async_copy %{{.*}} : memref<4x4xf32, strided<[2048, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
//          CHECK:   nvgpu.device_async_create_group %{{.*}} {__pipelining_first_stage__}
//
// CHECK:     scf.for %{{.*}} -> (!gpu.mma_matrix<16x16xf32, "COp">
//          CHECK:   nvgpu.device_async_wait %{{.*}} {numGroups = 4 : i32}
//          CHECK:   nvgpu.device_async_wait %{{.*}} {numGroups = 4 : i32}
//
//  CHECK-COUNT-8:   gpu.subgroup_mma_load_matrix %{{.*}} {leadDimension = 16 : index} : memref<64x16xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<16x8xf32, "AOp">
//  CHECK-COUNT-8:   gpu.subgroup_mma_load_matrix %{{.*}} {leadDimension = 128 : index} : memref<16x64xf32, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<8x16xf32, "BOp">
//
// CHECK-COUNT-32:   gpu.subgroup_mma_compute %{{.*}} : !gpu.mma_matrix<16x8xf32, "AOp">, !gpu.mma_matrix<8x16xf32, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
//
//  CHECK-COUNT-4:   nvgpu.device_async_copy %{{.*}} : memref<4x4xf32, strided<[1024, 1], offset: ?>> to memref<4x4xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
//          CHECK:   nvgpu.device_async_create_group %{{.*}} {__pipelining_first_stage__}
//          CHECK:   gpu.barrier {__pipelining_first_stage__}
//  CHECK-COUNT-4:   nvgpu.device_async_copy %{{.*}} : memref<4x4xf32, strided<[2048, 1], offset: ?>> to memref<4x4xf32, strided<[128, 1], offset: ?>, #gpu.address_space<workgroup>>
//          CHECK:   nvgpu.device_async_create_group %{{.*}} {__pipelining_first_stage__}
//          CHECK:   scf.yield %{{.*}} : !gpu.mma_matrix<16x16xf32, "COp">
//          CHECK: }
// CHECK-COUNT-16:     gpu.subgroup_mma_store_matrix %{{.*}} {leadDimension = 2048 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<64x64xf32, strided<[2048, 1], offset: ?>>
//          CHECK:     memref.dealloc %{{.*}} : memref<5x128x16xf32, #gpu.address_space<workgroup>>
//          CHECK:     memref.dealloc %{{.*}} : memref<5x16x128xf32, #gpu.address_space<workgroup>>

transform.sequence failures(propagate) {
^bb1(%variant_op: !transform.any_op):
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op

  // Step 1. Tile to forall and sequential scf.for.
  // ======================================================
  %forall_l1, %matmul_l1 =
    transform.structured.tile_to_forall_op %matmul tile_sizes [128, 128]
      ( mapping = [#gpu.block<y>, #gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  transform.iree.populate_workgroup_count_region_using_num_threads_slice
    %forall_l1 : (!transform.any_op) -> ()
  %fill_l1 = transform.structured.fuse_into_containing_op %fill into %forall_l1  : (!transform.any_op, !transform.any_op) -> !transform.any_op
  %matmul_l2, %loops:1 = transform.structured.tile_to_scf_for %matmul_l1 [0, 0, 16] : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  // // Post-tiling canonicalizations and cleanups.
  // transform.iree.apply_patterns %variant_op 
  //   {canonicalization, cse, licm, tiling_canonicalization}

  // Step 2. Pad the matmul and force packing to create the buffer in shared memory
  // Note: hoisting here may be dangerous memory-consumption-wise and we may be
  // better off with pipelining only.
  // ==============================================================================
  %matmul_padded_l2 = transform.structured.pad %matmul_l2 {
    padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32], 
    padding_dimensions = [0, 1, 2], 
    pack_paddings=[1, 1, 1]
  } : (!transform.any_op) -> !transform.any_op
  // Post-padding canonicalizations and cleanups.
  transform.iree.apply_patterns %variant_op 
    {canonicalization, cse, licm, tiling_canonicalization} : (!transform.any_op) -> ()

  // Step 3. Rewrite tensor.pad in DPS, this creates linalg.copy ops.
  // ================================================================
  %pad = transform.structured.match ops{["tensor.pad"]} in %variant_op 
    : (!transform.any_op) -> !transform.any_op
  %padded = transform.structured.rewrite_in_destination_passing_style %pad 
    : (!transform.any_op) -> !transform.any_op
  
  // Step 4. Map to threads, **SIMT** programming model.
  // ===================================================
  %copy_lhs = transform.get_producer_of_operand %matmul_padded_l2[0] 
     : (!transform.any_op) -> !transform.any_op
  transform.structured.tile_to_forall_op %copy_lhs num_threads [32, 4]
      ( mapping = [#gpu.linear<y>, #gpu.linear<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  %copy_rhs = transform.get_producer_of_operand %matmul_padded_l2[1]
     : (!transform.any_op) -> !transform.any_op
  transform.structured.tile_to_forall_op %copy_rhs num_threads [4, 32]
      ( mapping = [#gpu.linear<y>, #gpu.linear<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // %copy_res = transform.get_producer_of_operand %matmul_padded_l2[2]
  //    : (!transform.any_op) -> !transform.any_op
  // transform.structured.tile_to_forall_op %copy_res num_threads [4, 32]
  //     ( mapping = [#gpu.linear<y>, #gpu.linear<x>] )

  // Step 5. Contraction part mapped to threads with a **SIMD** programming model.
  // =============================================================================
  %forall_l3, %matmul_padded_l3 = 
    transform.structured.tile_to_forall_op %matmul_padded_l2 num_threads [2, 2]
      ( mapping = [#gpu.warp<y>, #gpu.warp<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %forall_fill_l3, %fill_l3 = 
    transform.structured.tile_to_forall_op %fill_l1 num_threads [2, 2]
      ( mapping = [#gpu.warp<y>, #gpu.warp<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

  // Step 6. Rank-reduce and vectorize.
  // ==================================
  %func_v = transform.structured.match ops{["func.func"]} in %variant_op
    : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_patterns %func_v { rank_reducing_linalg, rank_reducing_vector }
    : (!transform.any_op) -> ()
  %func_v_3 = transform.structured.vectorize %func_v { vectorize_padding }  : (!transform.any_op) -> !transform.any_op
  // Post-vectorization canonicalizations and hoistings to avoid roundtripping 
  // vectors in memory and prepare for bufferization.
  transform.iree.apply_patterns %func_v_3 {canonicalization, cse, licm }
    : (!transform.any_op) -> ()
  transform.structured.hoist_redundant_tensor_subsets %func_v_3
    : (!transform.any_op) -> ()

  // Step 7. Bufferize and drop HAL descriptor from memref ops.
  // ==========================================================
  // Pre-buferization canonicalizations and cleanups help avoid extra copies.
  transform.iree.apply_patterns %func_v_3 {canonicalization, cse, licm}
    : (!transform.any_op) -> ()
  transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op
    : (!transform.any_op) -> (!transform.any_op)
  %func_m = transform.structured.match ops{["func.func"]} in %variant_op_3 
    : (!transform.any_op) -> !transform.any_op
  transform.iree.erase_hal_descriptor_type_from_memref %func_m
    : (!transform.any_op) -> ()
  transform.iree.apply_buffer_optimizations %func_m : (!transform.any_op) -> ()

  // Step 8. Post-bufferization mapping blocks/workgroup and threads/subgroup.
  // =========================================================================
  transform.iree.apply_patterns %variant_op_3 
    {canonicalization, cse, licm, tiling_canonicalization}
    : (!transform.any_op) -> ()
  transform.iree.forall_to_workgroup %func_m : (!transform.any_op) -> ()
  transform.iree.map_nested_forall_to_gpu_threads %func_m
      workgroup_dims = [64, 2, 1] warp_dims = [2, 2, 1]
    : (!transform.any_op) -> ()

  //===---------------------------------------------------------------------===//
  // BEGIN - Annoying phase-ordered section
  //===---------------------------------------------------------------------===//
  // Vector transfer_read and transfer_write patterns have different subview
  // folding behavior, force a fold_memref_aliases on them to enable redundant
  // vector transfer hoisting.
  // Unfortunately, fold_memref_aliases breaks vector_to_mma conversion across 
  // scf.for after unrolling dur to insert_strided_slice / extract_strided_slice
  // across iter_args boundaries.
  // transform.iree.apply_patterns %func_m {canonicalization, cse, fold_memref_aliases}
  //   : (!transform.any_op) -> ()
  transform.iree.apply_patterns %func_m {canonicalization, cse, licm}
    : (!transform.any_op) -> ()
  transform.iree.hoist_static_alloc %func_m : (!transform.any_op) -> ()
  transform.iree.apply_patterns %func_m { fold_memref_aliases }
    : (!transform.any_op) -> ()
  transform.iree.apply_patterns %func_m { extract_address_computations }
    : (!transform.any_op) -> ()
  transform.iree.apply_patterns %func_m {canonicalization, cse, licm}
    : (!transform.any_op) -> ()
  transform.iree.apply_patterns %func_m { unroll_vectors_gpu_wmma }
    : (!transform.any_op) -> ()
    
  // Hoist redundant vector transfers to allow vectorization to proceed.
  // We really don't want to do this after bufferization but we need to atm.
  // One way to work around this is to hoist the pad ops on the output earlier 
  // but this has other tradeoffs. Still needs some investigation.
  %func_m_2 = transform.structured.hoist_redundant_vector_transfers %func_m
    : (!transform.any_op) -> !transform.any_op
  transform.iree.apply_buffer_optimizations %func_m_2 : (!transform.any_op) -> ()

  // This must occur after bufferization because of the fancy CUDA types.
  transform.iree.apply_patterns %func_m_2 { fold_memref_aliases }
    : (!transform.any_op) -> ()
  transform.iree.apply_patterns %func_m_2 {canonicalization, cse, licm}
    : (!transform.any_op) -> ()
  transform.iree.vector.vector_to_mma_conversion %func_m_2 { use_wmma }
    : (!transform.any_op) -> ()
  transform.iree.apply_patterns %func_m_2 {canonicalization, cse, licm}
    : (!transform.any_op) -> ()
  // //===---------------------------------------------------------------------===//
  // // END - Annoying phase-ordered section
  // //===---------------------------------------------------------------------===//

  // Step 9. Multi-buffering.
  // =========================================================================
  transform.iree.apply_patterns %func_m_2 {canonicalization, cse}
    : (!transform.any_op) -> ()
  // Hoist static allocs to allow multi-buffering to proceed.
  transform.iree.hoist_static_alloc %func_m_2 : (!transform.any_op) -> ()
  %allocs = transform.structured.match ops{["memref.alloc"]} in %func_m_2
    : (!transform.any_op) -> !transform.op<"memref.alloc">
  %mb_allocs = transform.memref.multibuffer %allocs {factor = 4 : i64, skip_analysis } 
    : (!transform.op<"memref.alloc">) -> !transform.any_op

  // Step 10. Cp-async.
  // ===========================================================================
  // Lower remaining vector ops to 1-D which will trigger the cp-async.
  // Alternatively we could explicitly unroll to 1-D innermost vectors if we 
  // wanted a specific target shape.
  %func_m_9 = transform.vector.transfer_to_scf %func_m_2
    max_transfer_rank = 1 full_unroll = true
      : (!transform.any_op) -> !transform.any_op
  transform.iree.create_async_groups %func_m_9 {use_mma_sync = false} 
    : (!transform.any_op) -> ()
  transform.iree.apply_patterns %func_m_9 {canonicalization, cse, fold_memref_aliases, licm}
    : (!transform.any_op) -> ()

  // Step 11. Pipeline shared memory copies.
  // ===========================================================================
  %mma_compute = transform.structured.match ops{["gpu.subgroup_mma_compute"]} in %variant_op_3
    : (!transform.any_op) -> !transform.any_op
  // Pre pipelining cleanups.
  transform.iree.apply_patterns %func_m_9 {canonicalization, cse}
    : (!transform.any_op) -> ()
  %for = transform.loop.get_parent_for %mma_compute : (!transform.any_op) -> !transform.op<"scf.for">
  %pipelined_for = transform.iree.pipeline_shared_memory_copies %for { depth = 4 } 
    : (!transform.op<"scf.for">) -> !transform.op<"scf.for">

  // Late canonicalizations and cleanups.
  transform.iree.apply_patterns %variant_op_3 
    {canonicalization, cse, licm, tiling_canonicalization}
    : (!transform.any_op) -> ()
}
