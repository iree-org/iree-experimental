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
//       --iree-codegen-llvmgpu-enable-transform-dialect-jit=false | \
//    FileCheck transform_dialect/examples/cuda/conv_2d_nchw_fchw_codegen_spec_step_01_wmma_ops.mlir
// ```

// CHECK: memref.alloc() {alignment = 64 : i64} : memref<16x16x1x1xf32, #gpu.address_space<workgroup>>
// CHECK: memref.alloc() {alignment = 64 : i64} : memref<16x16x1x1xf32, #gpu.address_space<workgroup>>
// CHECK: memref.alloc() {alignment = 64 : i64} : memref<16x16x1x1xf32, #gpu.address_space<workgroup>>
// CHECK: memref.alloc() {alignment = 64 : i64} : memref<3x3x16x16x1x1xf32, #gpu.address_space<workgroup>>
// CHECK: memref.alloc() {alignment = 64 : i64} : memref<3x16x16x1x1xf32, #gpu.address_space<workgroup>>
// CHECK: scf.for 
// CHECK:   scf.for 
// CHECK:     scf.for
// CHECK:       scf.for
// CHECK:         scf.for 
// CHECK:           nvgpu.device_async_copy {{.*}}, 1 : memref<16x16x132x132xf32> to memref<1x1x1x1xf32, strided<[16, 1, 1, 1], offset: ?>, #gpu.address_space<workgroup>>
// CHECK:           nvgpu.device_async_create_group %{{.*}}
// CHECK:           nvgpu.device_async_wait %{{.*}}
// CHECK:           gpu.barrier
//
// This is a spurious copy, get rid of it...
// CHECK:           linalg.generic {{.*}} ins(%{{.*}} : memref<16x16x1x1xf32, #gpu.address_space<workgroup>>) outs(%{{.*}} : memref<16x16x1x1xf32, strided<[16, 1, 1, 1], offset: ?>, #gpu.address_space<workgroup>>) {
// CHECK:         } 
// CHECK:         scf.for 
// CHECK:           scf.for
// CHECK:             nvgpu.device_async_copy {{.*}}, 1 : memref<64x16x3x3xf32> to memref<1x1x1x1xf32, strided<[16, 1, 1, 1], offset: ?>, #gpu.address_space<workgroup>>
// CHECK:             nvgpu.device_async_create_group %{{.*}}
// CHECK:             nvgpu.device_async_wait %{{.*}}
// CHECK:             gpu.barrier
//
// This is a spurious copy, get rid of it...
// CHECK:             linalg.generic 
// CHECK:           } 
// CHECK:         } 
// CHECK:         scf.for {{.*}} {
// CHECK:           nvgpu.device_async_copy %{{.*}}, 1 : memref<16x64x?x?xf32, strided<[1081600, 16900, 130, 1], offset: ?>> to memref<1x1x1x1xf32, strided<[16, 1, 1, 1], offset: ?>, #gpu.address_space<workgroup>>
// CHECK:           nvgpu.device_async_create_group %{{.*}}
// CHECK:           nvgpu.device_async_wait %{{.*}}
// CHECK:           gpu.barrier
// CHECK:           gpu.subgroup_mma_load_matrix %{{.*}} {leadDimension = 16 : index} : memref<16x16xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<16x8xf32, "AOp">
// CHECK:           gpu.subgroup_mma_load_matrix %{{.*}} {leadDimension = 16 : index} : memref<16x16xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<16x8xf32, "AOp">
// CHECK:           gpu.subgroup_mma_load_matrix %{{.*}} {leadDimension = 16 : index} : memref<16x16xf32, strided<[16, 1]>, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK:           gpu.subgroup_mma_load_matrix %{{.*}} {leadDimension = 16 : index, transpose} : memref<16x16xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<8x16xf32, "BOp">
// CHECK:           gpu.subgroup_mma_compute %{{.*}} : !gpu.mma_matrix<16x8xf32, "AOp">, !gpu.mma_matrix<8x16xf32, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK:           gpu.subgroup_mma_load_matrix %{{.*}} {leadDimension = 16 : index, transpose} : memref<16x16xf32, strided<[16, 1], offset: ?>, #gpu.address_space<workgroup>> -> !gpu.mma_matrix<8x16xf32, "BOp">
// CHECK:           gpu.subgroup_mma_compute %{{.*}} : !gpu.mma_matrix<16x8xf32, "AOp">, !gpu.mma_matrix<8x16xf32, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK:           gpu.subgroup_mma_store_matrix %{{.*}} {leadDimension = 16 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<16x16xf32, strided<[16, 1]>, #gpu.address_space<workgroup>>
//
// This is a spurious copy, get rid of it...
// CHECK:             linalg.generic 
// CHECK:    memref.dealloc %{{.*}} : memref<3x3x16x16x1x1xf32, #gpu.address_space<workgroup>>
// CHECK:    memref.dealloc %{{.*}} : memref<3x16x16x1x1xf32, #gpu.address_space<workgroup>>
// CHECK:    memref.dealloc %{{.*}} : memref<16x16x1x1xf32, #gpu.address_space<workgroup>>
// CHECK:    memref.dealloc %{{.*}} : memref<16x16x1x1xf32, #gpu.address_space<workgroup>>
// CHECK:    memref.dealloc %{{.*}} : memref<16x16x1x1xf32, #gpu.address_space<workgroup>>
// CHECK:    return

transform.structured.canonicalized_sequence failures(propagate) {
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
    // hoist padding memory usage may blow up memory without splitK
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
  %func_v_2 = transform.iree.apply_patterns %func_v { rank_reducing_linalg, rank_reducing_vector }
  %func_v_3 = transform.structured.vectorize %func_v_2
  %func_v_4 = transform.iree.apply_patterns %func_v_3 { unroll_vectors_gpu_wmma }

  // Step 6. Bufferize and drop HAL descriptor from memref ops.
  // ==========================================================
  %variant_op_2 = transform.iree.eliminate_empty_tensors %variant_op
  %variant_op_3 = transform.iree.bufferize { target_gpu } %variant_op_2
  %func_m = transform.structured.match ops{["func.func"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  %func_m_2 = transform.iree.erase_hal_descriptor_type_from_memref %func_m

  // This must occur after bufferization because of the fancy CUDA types.
  // TODO: Atm the slice does not get rewritten, seems brittle, figure this out.
  %func_m_3 = transform.iree.vector.vector_to_mma_conversion %func_m_2 { use_wmma }

  // Step 7. Post-bufferization mapping workgroup.
  // =============================================
  %func_m_4 = transform.iree.forall_to_workgroup %func_m_3
  %func_m_5 = transform.iree.map_nested_forall_to_gpu_threads %func_m_4
      { workgroup_size = [16, 16, 1] }
  %func_m_6 = transform.iree.apply_buffer_optimizations %func_m_5

  // Step 8. Multi-buffering, async copies and pipelining.
  // =========================================================================
  // We want to avoid blanket hoistings afer alloc hoisting, otherwise subviews
  // get hoisted and multibuffering fails because its preconditions are too 
  // fragile.
  // So we wrap in a transform.sequence for the moment.
  %func_m_7 = transform.cast %func_m_6 : !pdl.operation to !transform.op<"func.func">
  transform.sequence %func_m_7 : !transform.op<"func.func"> failures(propagate) {
  ^bb1(%func_arg: !transform.op<"func.func">):
    %func_m_8 = transform.iree.hoist_static_alloc %func_arg
      : (!transform.op<"func.func">) -> !transform.op<"func.func">
    %allocs = transform.structured.match ops{["memref.alloc"]} in %variant_op_3
      : (!pdl.operation) -> !transform.op<"memref.alloc">
    // TODO: In this particular case, mtuli-buffering does not work.
    // Figure it out later.
    %mb_allocs = transform.memref.multibuffer %allocs {factor = 2 : i64, skip_analysis} 
      : (!transform.op<"memref.alloc">) -> !pdl.operation
  }
  // Rewrite as cp.async.
  %func_a = transform.structured.match ops{["func.func"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  %func_a_2 = transform.iree.create_async_groups %func_a {use_mma_sync = false} 
    : (!pdl.operation) -> (!pdl.operation)
  // // Pipelining (must match the amount of multi-buffering).
  // // TODO: Matching the loop by return type is fragile here.
  // %for = transform.structured.match ops{["scf.for"]} 
  //   filter_result_type = !gpu.mma_matrix<16x16xf32, "COp"> in %variant_op_3 
  //   : (!pdl.operation) -> !transform.op<"scf.for">
  // %2 = transform.iree.pipeline_shared_memory_copies %for { depth = 2 } 
  //   : (!transform.op<"scf.for">) -> !transform.op<"scf.for">
}
