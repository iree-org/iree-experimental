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
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
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
//      --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_wip.mlir \
//      --iree-codegen-llvmgpu-enable-transform-dialect-jit=false | \
//   FileCheck transform_dialect/examples/cuda/matmul_wip.mlir
// ```
//
// To produce PTX:
// ```
//   export IREE_DIR=${HOME}/github/iree; 
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; 
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir | \
//   sed "s/\${M}/1024/g" | sed "s/\${K}/2048/g" | sed "s/\${N}/4096/g" | \
//   ${IREE_DIR}/build/tools/iree-compile - \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_wip.mlir \
//     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false 
// ```
//
// To run e2e on a remote machine (${A100_MACHINE}) with an A100 GPU:
// ```
//   # Do this only once:
//   # scp ${IREE_DIR}/build/tools/iree-run-module ${USER}@${A100_MACHINE}:~/;
//
//   export IREE_DIR=${HOME}/github/iree; 
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; 
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir | \
//   sed "s/\${M}/1024/g" | sed "s/\${K}/2048/g" | sed "s/\${N}/4096/g" | \
//   ${IREE_DIR}/build/tools/iree-compile - \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_wip.mlir \
//     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
//     --iree-hal-benchmark-dispatch-repeat-count=5 \
//     -o /tmp/foo.vmfb; \
//   scp /tmp/foo.vmfb ${USER}@${A100_MACHINE}:~/ > /dev/null; \
//   ssh ${USER}@${A100_MACHINE} "nsys profile --stats=true ~/iree-run-module --function=matmul_static --device=cuda --module=foo.vmfb --input=1024x2048xf32=1 --input=2048x4096xf32=1 --input=1024x4096xf32=1 2>&1" | \
//   grep matmul_static_dispatch | awk '{print $6}'
//
//   # The above prints the min across the 5 invocations.
//   # Alternatively, grep a little more to see what happens in more detail.
//   grep -3 matmul_static_dispatch
// ```
//
// The above command simply prints `10372752` (i.e. 10.37 million nanoseconds).

transform.sequence failures(propagate) {
// transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):

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
  %pad = transform.get_producer_of_operand %matmul_padded_l2[2]
    : (!pdl.operation) -> !pdl.operation
  %hoisted_pad = transform.structured.hoist_pad %pad by 1 loops
    : (!pdl.operation) -> !pdl.operation

  // Step 3. Rewrite tensor.pad in DPS. 
  // TODO: This must introduce unfoldable copies that disable the 
  // tensor::InsertSliceOp::fold very aggressive blanket behavior
  // ==============================================================
  %copy = transform.structured.rewrite_in_destination_passing_style %hoisted_pad
    : (!pdl.operation) -> !pdl.operation

  // Step 4. Map copies to warps, **SIMT** programming model.
  // Ensure we lower to 1-D vectors otherwise cp.async will not kick in.
  // ===================================================================
  transform.structured.tile_to_forall_op %copy num_threads [16, 4]
      ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )

  // Step 5. Map contraction to warps, **SIMD** programming model.
  // TODO: This step prevents hoisting C atm, which also breaks double-buffering.
  // This will be fixed a bit later, once a revamp of hoisting on tensors has 
  // landed.
  // ============================================================================
  %forall_l3, %matmul_padded_l3 = 
    transform.structured.tile_to_forall_op %matmul_padded_l2 num_threads [1]
      ( mapping = [#gpu.warp<x>] )

  // Step 6. Rank-reduce and vectorize.
  // ===================================================================================
  %func_v = transform.structured.match ops{["func.func"]} in %variant_op 
    : (!pdl.operation) -> !pdl.operation
  %func_v_2 = transform.iree.apply_patterns %func_v { rank_reducing_linalg, rank_reducing_vector }
  %func_v_3 = transform.structured.vectorize %func_v_2 { vectorize_padding }
  %func_v_4 = transform.iree.apply_patterns %func_v_3 { unroll_vectors_gpu_wmma }
  %func_v_5 = transform.structured.hoist_redundant_tensor_subsets %func_v_4
    : (!pdl.operation) -> !pdl.operation

  // Step 7. Bufferize and drop HAL descriptor from memref ops.
  // ==========================================================
  %variant_op_2 = transform.iree.eliminate_empty_tensors %variant_op
  %variant_op_3 = transform.iree.bufferize { allow_return_allocs, target_gpu } %variant_op_2
  %func_m = transform.structured.match ops{["func.func"]} in %variant_op_3
    : (!pdl.operation) -> !pdl.operation
  %func_m_2 = transform.iree.erase_hal_descriptor_type_from_memref %func_m

  // Step 8. Rewrite vectors as wmma operations.
  // ===========================================
  // This must occur after bufferization because of the fancy CUDA types.
  %func_m_3 = transform.iree.vector.vector_to_mma_conversion %func_m_2 { use_wmma }

  // Step 9. Post-bufferization mapping blocks/workgroup and threads/subgroup.
  // =========================================================================
  %func_m_4 = transform.iree.forall_to_workgroup %func_m_3
  %func_m_5 = transform.iree.map_nested_forall_to_gpu_threads %func_m_4
      { workgroup_size = [4, 16, 1] }
  %func_m_6 = transform.iree.apply_buffer_optimizations %func_m_5

  // Step 10. Multi-buffering, async copies and pipelining.
  // =========================================================================
  // We want to avoid blanket hoistings afer alloc hoisting, otherwise subviews
  // get hoisted and multibuffering fails because its preconditions are too 
  // fragile.
  // %func_m_7 = transform.cast %func_m_6 : !pdl.operation to !transform.op<"func.func">
  // transform.sequence %func_m_7 : !transform.op<"func.func"> failures(propagate) {
  // ^bb1(%func_arg: !transform.op<"func.func">):
    %func_m_7 = transform.cast %func_m_6: !pdl.operation to !transform.op<"func.func">
    %func_m_8 = transform.iree.hoist_static_alloc %func_m_7
      : (!transform.op<"func.func">) -> !transform.op<"func.func">
    %allocs = transform.structured.match ops{["memref.alloc"]} in %variant_op_3
      : (!pdl.operation) -> !transform.op<"memref.alloc">
    %mb_allocs = transform.memref.multibuffer %allocs {factor = 2 : i64, skip_analysis} 
      : (!transform.op<"memref.alloc">) -> !pdl.operation
  // }

  // Rewrite as cp.async.
  %func_a = transform.structured.match ops{["func.func"]} in %variant_op_3 
    : (!pdl.operation) -> !pdl.operation
  %func_a_2 = transform.iree.create_async_groups %func_a {use_mma_sync = false} 
    : (!pdl.operation) -> (!pdl.operation)

  // Pipelining (must match the amount of multi-buffering).
  // TODO: Matching the loop by return type is fragile here.
  %for = transform.structured.match ops{["scf.for"]} 
    filter_result_type = !gpu.mma_matrix<16x16xf32, "COp"> in %variant_op_3 
    : (!pdl.operation) -> !transform.op<"scf.for">
  %2 = transform.iree.pipeline_shared_memory_copies %for { depth = 2 } 
    : (!transform.op<"scf.for">) -> !transform.op<"scf.for">
}
