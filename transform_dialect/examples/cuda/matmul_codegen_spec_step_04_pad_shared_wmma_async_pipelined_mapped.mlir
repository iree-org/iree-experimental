// Instructions; TL;DR
// ===================
//
// This script shows an example of tiling for 2 levels with:
//   - padding with nofold to achieve shared memory mapping
//   - SIMT thread mapping of the copies to shared memory
//   - SIMD warp mapping of the contractions to shared memory
//   - connecting to wmma operations.
//   - creating async copies.
//
// The following are currently missing:
//   - hoisting of C post SIMD warp mapping.
//   - multibuffering.
//   - pipelining of async copies.
//
// These will be fixed once a revamp of hoisting on tensors lands upstream, which
// will allow us to host a padded version of C.
//
// Repro is dependent on landing https://github.com/nicolasvasilache/iree/tree/matmul-e2e
//
// TODO: the PTX generation currently fails with     
// ```
//    failed to legalize operation 'gpu.subgroup_mma_load_matrix' that was explicitly marked illegal
// ```
//
// ```
//   export IREE_DIR=/usr/local/google/home/ntv/github/iree; \
//   export IREE_SAMPLES_DIR=/usr/local/google/home/ntv/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
//   sed "s/\${M}/1024/g" | sed "s/\${K}/2048/g" | sed "s/\${N}/4096/g" | \
//   ${IREE_DIR}/build/tools/iree-opt \
//     --iree-hal-target-backends=cuda \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_DIR}/build/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
//      --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_codegen_spec_step_04_pad_shared_wmma_async_pipelined_mapped.mlir \
//      --iree-codegen-llvmgpu-enable-transform-dialect-jit=false | \
//   FileCheck transform_dialect/examples/cuda/matmul_codegen_spec_step_04_pad_shared_wmma_async_pipelined_mapped.mlir
// ```
//
// To produce PTX:
// ```
//   export IREE_DIR=/usr/local/google/home/ntv/github/iree; 
//   export IREE_SAMPLES_DIR=/usr/local/google/home/ntv/github/iree-samples; 
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir | \
//   sed "s/\${M}/1024/g" | sed "s/\${K}/2048/g" | sed "s/\${N}/4096/g" | \
//   ${IREE_DIR}/build/tools/iree-compile - \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_codegen_spec_step_04_pad_shared_wmma_async_pipelined_mapped.mlir \
//     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false 
// ```
//
// CHECK: memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #gpu.address_space<workgroup>>
// CHECK: memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #gpu.address_space<workgroup>>
// CHECK: hal.interface.workgroup.id[1] : index
// CHECK: hal.interface.workgroup.id[0] : index
// CHECK: gpu.thread_id  x
// CHECK: gpu.thread_id  y
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     scf.for
// CHECK:       nvgpu.device_async_copy %{{.*}}, 4 : memref<1024x2048xf32> to memref<1x4xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
// CHECK:       nvgpu.device_async_create_group %{{.*}}
// CHECK:       nvgpu.device_async_wait %{{.*}}
// CHECK:       gpu.barrier
// CHECK:       nvgpu.device_async_copy %{{.*}}, 4 : memref<2048x4096xf32> to memref<1x4xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
// CHECK:       nvgpu.device_async_create_group %{{.*}}
// CHECK:       nvgpu.device_async_wait %{{.*}}
// CHECK:       gpu.barrier
// CHECK:       gpu.subgroup_mma_load_matrix %{{.*}} {leadDimension = 16 : index} : memref<16x16xf32, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<8x16xf32, "AOp">
// CHECK:       gpu.subgroup_mma_load_matrix %{{.*}} {leadDimension = 16 : index} : memref<16x16xf32, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<16x16xf32, "BOp">
// CHECK:       gpu.subgroup_mma_load_matrix %{{.*}} {leadDimension = 4096 : index} : memref<16x16xf32, strided<[4096, 1], offset: ?>> -> !gpu.mma_matrix<8x16xf32, "COp">
// CHECK:       gpu.subgroup_mma_compute %{{.*}} : !gpu.mma_matrix<8x16xf32, "AOp">, !gpu.mma_matrix<16x16xf32, "BOp"> -> !gpu.mma_matrix<8x16xf32, "COp">
// CHECK:       gpu.subgroup_mma_store_matrix %{{.*}} {leadDimension = 4096 : index} : !gpu.mma_matrix<8x16xf32, "COp">, memref<8x16xf32, strided<[4096, 1], offset: ?>>
// CHECK:       gpu.barrier
// CHECK: memref.dealloc %{{.*}} : memref<16x16xf32, #gpu.address_space<workgroup>>
// CHECK: memref.dealloc %{{.*}} : memref<16x16xf32, #gpu.address_space<workgroup>>
// CHECK: return

// transform.structured.canonicalized_sequence failures(propagate) {
transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):

  // %func = transform.structured.match ops{["func.func"]} in %variant_op
  //   : (!pdl.operation) -> !pdl.operation
  // %func_2 = transform.cast %func : !pdl.operation to !transform.op<"func.func">

  // transform.sequence %func_2 : !transform.op<"func.func"> failures(propagate) {
  //   ^bb1(%func_arg: !transform.op<"func.func">):

  //   %matmul = transform.structured.match ops{["linalg.matmul"]} in %func_arg
  //     : (!transform.op<"func.func">) -> (!pdl.operation)

  //   // Step 1. Tile to forall and sequential scf.for.
  //   // ======================================================
  //   %forall_l1, %matmul_l1 =
  //     transform.iree.tile_to_forall_and_workgroup_count_region %matmul tile_sizes [128, 128]
  //       ( mapping = [#gpu.block<y>, #gpu.block<x>] )
  //   %matmul_l2, %loops:3 = transform.structured.tile_to_scf_for %matmul_l1 [16, 16, 16]

  //   // Step 2. Pad the matmul and force packing to create the buffer in shared memory
  //   // Note: hoisting here may be dangerous memory-consumption-wise and we may be
  //   // better off with pipelining only.
  //   // ==============================================================================
  //   %matmul_padded_l2 = transform.structured.pad %matmul_l2 {
  //     padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32], 
  //     padding_dimensions = [0, 1, 2], 
  //     pack_paddings=[0, 0, 1]
  //     // hoist padding memory usage may blow up memory without splitK
  //     , hoist_paddings=[0, 0, 1]
  //   }
  // }


  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
     : (!pdl.operation) -> !pdl.operation

  // Step 1. Tile to forall and sequential scf.for.
  // ======================================================
  %forall_l1, %matmul_l1 =
    transform.iree.tile_to_forall_and_workgroup_count_region %matmul tile_sizes [128, 128]
      ( mapping = [#gpu.block<y>, #gpu.block<x>] )
  %matmul_l2, %loops:3 = transform.structured.tile_to_scf_for %matmul_l1 [16, 16, 16]
  
  // Step 2. Pad the matmul and force packing to create the buffer in shared memory
  // Note: hoisting here may be dangerous memory-consumption-wise and we may be
  // better off with pipelining only.
  // ==============================================================================
  %matmul_padded_l2 = transform.structured.pad %matmul_l2 {
    padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32], 
    padding_dimensions = [0, 1, 2], 
    pack_paddings=[0, 0, 1]
  }

  %pad = transform.structured.match ops{["tensor.pad"]} in %variant_op 
    : (!pdl.operation) -> !pdl.operation
  %padded = transform.structured.hoist_pad %pad by 1 loops
    : (!pdl.operation) -> !transform.op<"tensor.pad">

  // %func = transform.structured.match ops{["func.func"]} in %variant_op
  //   : (!pdl.operation) -> !pdl.operation
  // transform.structured.hoist_redundant_tensor_subsets %func
  //   : (!pdl.operation) -> !pdl.operation

  // From now on, we want a canonicalized_sequence.
  transform.structured.canonicalized_sequence %variant_op failures(propagate) {
    ^bb1(%inner_variant_op: !pdl.operation):
 
    // Step 3. Rewrite tensor.pad in DPS. 
    // TODO: This must introduce unfoldable copies that disable the 
    // tensor::InsertSliceOp::fold very aggressive blanket behavior
    // ==============================================================
    %pad_2 = transform.structured.match ops{["tensor.pad"]} in %inner_variant_op 
      : (!pdl.operation) -> !pdl.operation
    %padded_2 = transform.structured.rewrite_in_destination_passing_style %pad_2
      : (!pdl.operation) -> !pdl.operation
    
    // Step 4. Map copies to threads, **SIMT** programming model.
    // Ensure we lower to 1-D vectors otherwise cp.async will not kick in.
    // ===================================================================
    %fill = transform.structured.match ops{["linalg.fill"]} in %inner_variant_op
      : (!pdl.operation) -> !pdl.operation
    transform.structured.tile_to_forall_op %fill num_threads [16, 4]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )
    %copy = transform.structured.match ops{["linalg.copy"]} in %inner_variant_op
      : (!pdl.operation) -> !pdl.operation
    transform.structured.tile_to_forall_op %copy num_threads [16, 4]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )

    // Step 5. Map contraction to warps, **SIMD** programming model.
    // TODO: This step prevents hoisting C atm, which also breaks double-buffering.
    // This will be fixed a bit later, once a revamp of hoisting on tensors has 
    // landed.
    // ============================================================================
    %inner_matmul_padded_l2 = transform.structured.match ops{["linalg.matmul"]} in %inner_variant_op
      : (!pdl.operation) -> !pdl.operation
    %forall_l3, %matmul_padded_l3 = 
      transform.structured.tile_to_forall_op %inner_matmul_padded_l2 num_threads [2, 0]
        ( mapping = [#gpu.warp<x>] )

    // Step 6. Rank-reduce and vectorize.
    // ===================================================================================
    %func_v = transform.structured.match ops{["func.func"]} in %inner_variant_op 
      : (!pdl.operation) -> !pdl.operation
    %func_v_2 = transform.iree.apply_patterns %func_v { rank_reducing_linalg, rank_reducing_vector }
    %func_v_3 = transform.structured.vectorize %func_v_2 { vectorize_padding }
    %func_v_4 = transform.iree.apply_patterns %func_v_3 { unroll_vectors_gpu_wmma }

    // Step 7. Bufferize and drop HAL descriptor from memref ops.
    // ==========================================================
    // %inner_variant_op_2 = transform.iree.eliminate_empty_tensors %inner_variant_op
    %inner_variant_op_2 = transform.iree.bufferize 
      { allow_return_allocs, target_gpu } %inner_variant_op
    // %func_m = transform.structured.match ops{["func.func"]} in %inner_variant_op_2
    //   : (!pdl.operation) -> !pdl.operation
    // %func_m_2 = transform.iree.erase_hal_descriptor_type_from_memref %func_m

    // // Step 8. Rewrite vectors as wmma operations.
    // // ===========================================
    // // This must occur after bufferization because of the fancy CUDA types.
    // %func_m_3 = transform.iree.vector.vector_to_mma_conversion %func_m_2 { use_wmma }

    // // Step 9. Post-bufferization mapping blocks/workgroup and threads/subgroup.
    // // =========================================================================
    // %func_m_4 = transform.iree.forall_to_workgroup %func_m_3
    // %func_m_5 = transform.iree.map_nested_forall_to_gpu_threads %func_m_4
    //     { workgroup_size = [4, 16, 1] }
    // %func_m_6 = transform.iree.apply_buffer_optimizations %func_m_5

    // // Step 10. Multi-buffering, async copies and pipelining.
    // // =========================================================================
    // // We want to avoid blanket hoistings afer alloc hoisting, otherwise subviews
    // // get hoisted and multibuffering fails because its preconditions are too 
    // // fragile.
    // // So we wrap in a transform.sequence for the moment.
    // %func_m_7 = transform.cast %func_m_6 : !pdl.operation to !transform.op<"func.func">
    // transform.sequence %func_m_7 : !transform.op<"func.func"> failures(propagate) {
    // ^bb1(%func_arg: !transform.op<"func.func">):
    //   %func_m_8 = transform.iree.hoist_static_alloc %func_arg
    //     : (!transform.op<"func.func">) -> !transform.op<"func.func">
    //   %allocs = transform.structured.match ops{["memref.alloc"]} in %inner_variant_op_2
    //     : (!pdl.operation) -> !transform.op<"memref.alloc">
    //   %mb_allocs = transform.memref.multibuffer %allocs {factor = 2 : i64, skip_analysis} 
    //     : (!transform.op<"memref.alloc">) -> !pdl.operation
    // }
    // // Rewrite as cp.async.
    // %func_a = transform.structured.match ops{["func.func"]} in %inner_variant_op_2 
    //   : (!pdl.operation) -> !pdl.operation
    // %func_a_2 = transform.iree.create_async_groups %func_a {use_mma_sync = false} 
    //   : (!pdl.operation) -> (!pdl.operation)
    // // Pipelining (must match the amount of multi-buffering).
    // // TODO: Matching the loop by return type is fragile here.
    // %for = transform.structured.match ops{["scf.for"]} 
    //   filter_result_type = !gpu.mma_matrix<16x16xf32, "COp"> in %inner_variant_op_2 
    //   : (!pdl.operation) -> !transform.op<"scf.for">
    // %2 = transform.iree.pipeline_shared_memory_copies %for { depth = 2 } 
    //   : (!transform.op<"scf.for">) -> !transform.op<"scf.for">
  }
}
