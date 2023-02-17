// Instructions; TL;DR
// ===================
//
// This script shows an example of tiling for 2 levels with:
//   - padding with nofold to achieve shared memory mapping
//   - thread mapping of the copies to shared memory
//   - connecting to wmma operations.
// This is purely for illustration purposes as this does not perform any 
// thread/warp level mapping of the wmma operations.
//
// This is dependent on the integration of the prerequisites in:
//   https://github.com/nicolasvasilache/iree/commits/matmul-cuda-pad
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
//      --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_codegen_spec_step_02_pad_shared_wmma.mlir \
//      --iree-codegen-llvmgpu-enable-transform-dialect-jit=false | \
//   FileCheck transform_dialect/examples/cuda/matmul_codegen_spec_step_02_pad_shared_wmma.mlir
// ```

// CHECK: hal.interface.workgroup.id[1] : index
// CHECK: hal.interface.workgroup.id[0] : index
// CHECK: gpu.thread_id  x
// CHECK: gpu.thread_id  y
// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     gpu.subgroup_mma_load_matrix {{.*}} -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK:     scf.for {{.*}} -> (!gpu.mma_matrix<16x16xf32, "COp">) {
// CHECK:       memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #gpu.address_space<workgroup>>
// CHECK:       memref.subview %{{.*}} : memref<16x16xf32, #gpu.address_space<workgroup>> to memref<2x4xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
// CHECK:       vector.transfer_read %{{.*}} : memref<1024x2048xf32>, vector<2x4xf32>
// CHECK:       vector.transfer_write %{{.*}} : vector<2x4xf32>, memref<2x4xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
// CHECK:       gpu.barrier
// CHECK:       memref.alloc() {alignment = 64 : i64} : memref<16x16xf32, #gpu.address_space<workgroup>>
// CHECK:       memref.subview %{{.*}} : memref<16x16xf32, #gpu.address_space<workgroup>> to memref<2x4xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
// CHECK:       vector.transfer_read %{{.*}} : memref<2048x4096xf32>, vector<2x4xf32>
// CHECK:       vector.transfer_write %{{.*}} : vector<2x4xf32>, memref<2x4xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>>
// CHECK:       gpu.barrier
// CHECK:       gpu.subgroup_mma_load_matrix %{{.*}} {leadDimension = 16 : index} : memref<16x16xf32, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<16x8xf32, "AOp">
// CHECK:       gpu.subgroup_mma_load_matrix %{{.*}} {leadDimension = 16 : index} : memref<16x16xf32, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<16x8xf32, "AOp">
// CHECK:       gpu.subgroup_mma_load_matrix %{{.*}} {leadDimension = 16 : index} : memref<16x16xf32, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<8x16xf32, "BOp">
// CHECK:       gpu.subgroup_mma_load_matrix %{{.*}} {leadDimension = 16 : index} : memref<16x16xf32, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<8x16xf32, "BOp">
// CHECK:       gpu.subgroup_mma_compute %{{.*}} : !gpu.mma_matrix<16x8xf32, "AOp">, !gpu.mma_matrix<8x16xf32, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK:       gpu.subgroup_mma_compute %{{.*}} : !gpu.mma_matrix<16x8xf32, "AOp">, !gpu.mma_matrix<8x16xf32, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK:       memref.dealloc %{{.*}} : memref<16x16xf32, #gpu.address_space<workgroup>>
// CHECK:       memref.dealloc %{{.*}} : memref<16x16xf32, #gpu.address_space<workgroup>>
// CHECK:     gpu.subgroup_mma_store_matrix %{{.*}} {leadDimension = 4096 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<128x128xf32, strided<[4096, 1], offset: ?>>

transform.structured.canonicalized_sequence failures(propagate) {
// transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  // Step 1. Tile to foreach_thread and sequential scf.for.
  // ======================================================
  %foreach_thread_l1, %matmul_l1 =
    transform.iree.tile_to_foreach_thread_and_workgroup_count_region %matmul tile_sizes [128, 128]
      ( mapping = [#gpu.block<y>, #gpu.block<x>] )
  %matmul_l2, %loops:3 = transform.structured.tile_to_scf_for %matmul_l1 [16, 16, 16]

  // Step 2. Pad the matmul and force packing to create the buffer in shared memory
  // Note: hoisting here may be dangerous memory-consumption-wise and we may be
  // better off with pipelining only.
  // ==============================================================================
  %matmul_padded_l2 = transform.structured.pad %matmul_l2 {
    padding_values = [0.0 : f32, 0.0 : f32, 0.0 : f32], 
    padding_dimensions = [0, 1, 2], 
    pack_paddings=[1, 1, 0]
    // hoist padding memory usage may blow up memory without splitK
    // , hoist_paddings=[1, 1, 0]
  }

  // Step 3. Promote buffers.
  // TODO: This is not yet enough to expose linalg.copy on tensors and cannot yet
  // be used to tile to scf.foreach_thread.
  // ==============================================================
  // %promoted_matmul_l2, %alloc_1_op , %alloc_2_op = transform.iree.promote_operands %matmul_padded_l2 [0, 1] 
  //   : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)
  // %alloc_1 = transform.get_result %alloc_1_op[0] : (!pdl.operation) -> !transform.any_value
  // %alloc_1_buffer = transform.structured.bufferize_to_allocation %alloc_1 {memory_space = 3}

  // Step 3. Rewrite tensor.pad in DPS. 
  // TODO: This must introduce unfoldable copies that disable the 
  // tensor::InsertSliceOp::fold very aggressive blanket behavior
  // ==============================================================
  %pad = transform.structured.match ops{["tensor.pad"]} in %variant_op 
    : (!pdl.operation) -> !pdl.operation
  %padded = transform.structured.rewrite_in_destination_passing_style %pad 
    : (!pdl.operation) -> !pdl.operation
  
  // Step 4. Map to threads, **SIMT** programming model.
  // ===================================================
  %fill = transform.structured.match ops{["linalg.fill"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  transform.structured.tile_to_foreach_thread_op %fill num_threads [8, 4]
      ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )
  %copy = transform.structured.match ops{["linalg.copy"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  transform.structured.tile_to_foreach_thread_op %copy num_threads [8, 4]
      ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )

  // TODO: Contraction part must be mapped to threads with a **SIMD** programming model.
  // ===================================================================================

  // Step 5. Rank-reduce and vectorize.
  // ==================================
  %func_v = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %func_v_2 = transform.iree.apply_patterns %func_v { rank_reducing_linalg, rank_reducing_vector }
  %func_v_3 = transform.structured.vectorize %func_v_2 { vectorize_padding }
  %func_v_4 = transform.iree.apply_patterns %func_v_3 { unroll_vectors_gpu_mma }

  // Step 6. Bufferize and drop HAL descriptor from memref ops.
  // ==========================================================
  %variant_op_2 = transform.iree.eliminate_empty_tensors %variant_op
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op_2
  %func_m = transform.structured.match ops{["func.func"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  %func_m_2 = transform.iree.erase_hal_descriptor_type_from_memref %func_m

  // This must occur after bufferization because of the fancy CUDA types.
  %func_m_3 = transform.iree.vector.vector_to_mma_conversion %func_m_2

  // Step 7. Post-bufferization mapping blocks/workgroup and threads/subgroup.
  // =========================================================================
  %func_m_4 = transform.iree.foreach_thread_to_workgroup %func_m_3
  %func_m_5 = transform.iree.map_nested_foreach_thread_to_gpu_threads %func_m_4
      { workgroup_size = [4, 8, 1] }
  %func_m_6 = transform.iree.apply_buffer_optimizations %func_m_5

  %allocs = transform.structured.match ops{["memref.alloc"]} in %variant_op_3
    : (!pdl.operation) -> !transform.op<"memref.alloc">
  // TODO: this currently fails to multi-bufferize.
  // %mb_allocs = transform.memref.multibuffer %allocs {factor = 2 : i64} 
  //   : (!transform.op<"memref.alloc">) -> !pdl.operation

  //   %func_a = transform.structured.match ops{["func.func"]} in %variant_op 
  //     : (!pdl.operation) -> !pdl.operation
  //   %func_a_2 = transform.iree.create_async_groups %func_a {use_mma_sync = true} 
  //     : (!pdl.operation) -> (!pdl.operation)

  //   %for = transform.structured.match ops{["scf.for"]} 
  //     filter_result_type = !gpu.mma_matrix<16x16xf32, "COp"> in %variant_op 
  //     : (!pdl.operation) -> !transform.op<"scf.for">
  //   %2 = transform.iree.pipeline_shared_memory_copies %for { depth = 2 } 
  //     : (!transform.op<"scf.for">) -> !transform.op<"scf.for">
  //   transform.iree.apply_patterns %func_a_2 { canonicalize }
}
