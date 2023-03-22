// Instructions; TL;DR
// ===================
//
// Simple e2e example that demonstrates how to use the transform dialect to map
// a reduction to a GPU.
// There are currently a few issues with this example:
//   - we are missing any warp distribution of the reduction
//   - bufferization allocates when it should be inplace
//   - we are missing canonicalizations, cse, licm etc
//   - we are missing the lowering to NVVM
//
// ```
//     export IREE_DIR=${HOME}/github/iree; \
//     export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//     cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/reduction.mlir |\
//     sed "s/\${SZ1}/1024/g" | sed "s/\${SZ2}/9999/g" | sed "s/\${SZ3}/1024/g" | \
//     sed "s/\${ELEMENTAL_TYPE}/f32/g" | sed "s/\${ZERO}/0.0/g" | sed "s/\${ADD_OP}/arith.addf/g" | sed "s/\${DIV_OP}/arith.divf/g" | \
//     sed "s/private @reduction_2d_static(/@reduction_2d_static(/g" | \
//     ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//     ${LLVM_BUILD_DIR}/bin/mlir-opt \
//       --pass-pipeline="builtin.module(test-transform-dialect-interpreter{\
//         transform-file-name=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/reduction_step_01.mlir})"
/// ```

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %fill = transform.structured.match ops{["linalg.fill"]} in %module_op
    : (!pdl.operation) -> !pdl.operation
  %generic = transform.structured.match ops{["linalg.generic"]} in %module_op
    : (!pdl.operation) -> !pdl.operation

  // Step 1. Tile to forall and fuse.
  // ============================================================================
  %forall_l1, %generic_l1 =
    transform.structured.tile_to_forall_op %generic tile_sizes [1]
      ( mapping = [#gpu.block<x>] )
  %fill_l1 = transform.structured.fuse_into_containing_op %fill into %forall_l1

  // Step 2. Tile reduction to forall, which also pads on the fly.
  // ============================================================================
  %loop, %1, %2, %3 = transform.structured.tile_reduction_using_forall 
    %generic_l1 by num_threads = [0, 64], tile_sizes = [0, 4], mapping = [#gpu.thread<x>]

  // Step 3. Vectorize.
  // ============================================================================
  %func = transform.structured.match ops{["func.func"]} in %module_op 
    : (!pdl.operation) -> !pdl.operation
  %func_2 = transform.structured.vectorize %func

  // Step 4. Bufferize.
  // ============================================================================
  // TODO: eliminate tensor.empty ops and make sure we bufferize inplace.
  %empty_tensor = transform.structured.match ops{["tensor.empty"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.empty">
  transform.bufferization.empty_tensor_to_alloc_tensor %empty_tensor
    : (!transform.op<"tensor.empty">) -> !transform.op<"bufferization.alloc_tensor">
  transform.bufferization.one_shot_bufferize layout{IdentityLayoutMap} %module_op 
    {bufferize_function_boundaries = true, allow_return_allocs = true}
  %func_3 = transform.structured.vectorize %func_2

  // TODO: warp shuffles.

  %gpu_launch = transform.gpu.map_forall_to_blocks %func_3
    grid_dims = [1024] {generate_gpu_launch}
  transform.gpu.map_nested_forall_to_threads %gpu_launch 
    block_dims = [64, 1, 1]

  %func_e = transform.structured.match ops{["func.func"]} in %module_op 
    : (!pdl.operation) -> !pdl.operation
  transform.vector.lower_vectors %func_e multireduction_lowering = "innerreduction"

  // TODO: lower to NVVM
}
