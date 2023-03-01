// Instructions; TL;DR
// ===================
//
// This script shows a simple example of tiling for 2 levels and connecting to 
// wmma operations.
// This is purely for illustration purposes as this does not perform any 
// thread/warp level mapping or shared memory.
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
//   sed "s/\${M}/1024/g" | sed "s/\${N}/4096/g" | sed "s/\${K}/2048/g" | \
//   ${IREE_DIR}/build/tools/iree-opt \
//     --iree-hal-target-backends=cuda \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_DIR}/build/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
//      --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_codegen_spec_step_01_wmma_ops.mlir \
//      --iree-codegen-llvmgpu-enable-transform-dialect-jit=false | \
//   FileCheck transform_dialect/examples/cuda/matmul_codegen_spec_step_01_wmma_ops.mlir
// ```

// CHECK: scf.for
// CHECK:   scf.for
// CHECK:     gpu.subgroup_mma_load_matrix {{.*}} -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK:     scf.for {{.*}} -> (!gpu.mma_matrix<16x16xf32, "COp">) {
// CHECK:       gpu.subgroup_mma_load_matrix {{.*}} {leadDimension = 2048 : index} : memref<1024x2048xf32> -> !gpu.mma_matrix<16x8xf32, "AOp">
// CHECK:       gpu.subgroup_mma_load_matrix {{.*}} {leadDimension = 2048 : index} : memref<1024x2048xf32> -> !gpu.mma_matrix<16x8xf32, "AOp">
// CHECK:       gpu.subgroup_mma_load_matrix {{.*}} {leadDimension = 4096 : index} : memref<2048x4096xf32> -> !gpu.mma_matrix<8x16xf32, "BOp">
// CHECK:       gpu.subgroup_mma_load_matrix {{.*}} {leadDimension = 4096 : index} : memref<2048x4096xf32> -> !gpu.mma_matrix<8x16xf32, "BOp">
// CHECK:       gpu.subgroup_mma_compute {{.*}} : !gpu.mma_matrix<16x8xf32, "AOp">, !gpu.mma_matrix<8x16xf32, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK:       gpu.subgroup_mma_compute {{.*}} : !gpu.mma_matrix<16x8xf32, "AOp">, !gpu.mma_matrix<8x16xf32, "BOp"> -> !gpu.mma_matrix<16x16xf32, "COp">
// CHECK:       scf.yield {{.*}} : !gpu.mma_matrix<16x16xf32, "COp">
// CHECK:     gpu.subgroup_mma_store_matrix {{.*}} {leadDimension = 4096 : index} : !gpu.mma_matrix<16x16xf32, "COp">, memref<128x128xf32, strided<[4096, 1], offset: ?>>
transform.structured.canonicalized_sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!pdl.operation) -> !pdl.operation

  // Step 1. Tile to forall and sequential scf.for.
  // ======================================================
  %forall_l1, %matmul_l1 =
    transform.iree.tile_to_forall_and_workgroup_count_region %matmul tile_sizes [128, 128]
      ( mapping = [#gpu.block<y>, #gpu.block<x>] )
  %matmul_l2, %loops:3 = transform.structured.tile_to_scf_for %matmul_l1 [16, 16, 16]

  
  // Step 2. Rank-reduce and vectorize.
  // ==================================
  %func_v = transform.structured.match ops{["func.func"]} in %variant_op : (!pdl.operation) -> !pdl.operation
  %func_v_2 = transform.iree.apply_patterns %func_v { rank_reducing_linalg, rank_reducing_vector }
  %func_v_3 = transform.structured.vectorize %func_v_2
  %func_v_4 = transform.iree.apply_patterns %func_v_3 { unroll_vectors_gpu_wmma }

  // Step 3. Bufferize and drop HAL descriptor from memref ops.
  // ==========================================================
  %variant_op_2 = transform.iree.eliminate_empty_tensors %variant_op
  %variant_op_3 = transform.iree.bufferize %variant_op_2
  %func_m = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!pdl.operation) -> !pdl.operation
  %func_m_2 = transform.iree.erase_hal_descriptor_type_from_memref %func_m

  // This must occur after bufferization because of the fancy CUDA types.
  %func_m_3 = transform.iree.vector.vector_to_mma_conversion %func_m_2 { use_wmma }

  // Step 3. Post-bufferization mapping workgroup.
  // =============================================
  transform.iree.forall_to_workgroup %func_m_3
}
