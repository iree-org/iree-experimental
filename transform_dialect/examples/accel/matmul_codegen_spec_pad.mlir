// This script shows an example lowering matmul through IREE for a special accelerator.
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   ${IREE_DIR}/build/tools/iree-opt \
//     ${IREE_SAMPLES_DIR}/transform_dialect/examples/accel/matmul_source.mlir \
//     --iree-hal-target-backends=llvm-cpu \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_DIR}/build/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target, builtin.module(func.func(iree-hoist-statically-bound-allocations)))))' \
//      --iree-codegen-llvmcpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/accel/matmul_codegen_spec_pad.mlir
// ```

module attributes { transform.with_named_sequence } {
  transform.named_sequence @cleanup(%variant_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %func {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %func : !transform.any_op
    transform.iree.apply_cse %func : !transform.any_op
    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // First level tile to forall with tile_sizes [16, 32].
    %forall, %tiled_matmul =
      transform.structured.tile_to_forall_op %matmul tile_sizes [16, 32]
        ( mapping = [#gpu.block<y>, #gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall
      : (!transform.any_op) -> ()

    // Tile reduction dimension.
    %tiled_reduction, %loop =
      transform.structured.tile %tiled_matmul [0, 0, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pad operation.
    %padded, %pad, %__ = transform.structured.pad %tiled_reduction {
      padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[1, 1, 0],
      copy_back_op="none"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Second level tile to forall with tile_sizes [8, 8].
    %forall_1, %tiled_matmul_1 =
      transform.structured.tile_to_forall_op %padded tile_sizes [8, 8]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pad operation.
    %padded_1, %pad_1, %_ = transform.structured.pad %tiled_matmul_1 {
      padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[0, 0, 1],
      copy_back_op="none"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()

    // Bufferize and drop HAL decriptor from memref ops.
    %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> !transform.any_op

    // Post-bufferization mapping workgroup.
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %memref_func workgroup_dims = [4, 2, 1] subgroup_size = 1 : (!transform.any_op) -> ()
    transform.iree.hoist_static_alloc %memref_func : (!transform.any_op) -> ()
  }
}
