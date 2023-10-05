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

    // First level tile to forall with tile_sizes [16, 128].
    %forall, %tiled_matmul =
      transform.structured.tile_to_forall_op %matmul tile_sizes [16, 128]
        ( mapping = [#gpu.block<y>, #gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall
      : (!transform.any_op) -> ()

    // Tile reduction dimension.
    %tiled_reduction, %loop =
      transform.structured.tile %tiled_matmul [0, 0, 128]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pack by applying data tiling, and the linalg.matmul becomes linalg.mmt4d.
    %packed = transform.structured.pack %tiled_reduction packed_sizes = [16, 64, 64]
      : (!transform.any_op) -> (!transform.any_op)

    // Transpose B matrix from [K N n k] to [K N k n]
    %pack_producer_b0 = transform.get_producer_of_operand %packed[1]
      : (!transform.any_op) -> (!transform.any_op)
    %packed_b0, %pack_b0, %empty_unpack_b0 =
      transform.structured.pack_transpose %pack_producer_b0 with_compute_op(%packed)
      inner_perm = [1, 0] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Second level tile to forall with tile_sizes [16, 64].
    %forall_1, %tiled_matmul_1 =
      transform.structured.tile_to_forall_op %packed_b0 tile_sizes [16, 64]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pack by applying data tiling, and the linalg.matmul becomes linalg.mmt4d.
    %packed_2 = transform.structured.pack %tiled_matmul_1 packed_sizes = [0, 0, 0, 4, 8, 8]
      : (!transform.any_op) -> (!transform.any_op)

    // Transpose A matrix from [M K m k m0 k0] to [M K k m m0 k0]
    %pack_producer_a = transform.get_producer_of_operand %packed_2[0]
      : (!transform.any_op) -> (!transform.any_op)
    %packed_a, %pack_a, %empty_unpack_a =
      transform.structured.pack_transpose %pack_producer_a with_compute_op(%packed_2)
      outer_perm = [0, 1, 3, 2] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Transpose B matrix from [K N k n n0 k0] to [K N n k k0 n0]
    %pack_producer_b = transform.get_producer_of_operand %packed_a[1]
      : (!transform.any_op) -> (!transform.any_op)
    %packed_b, %pack_b, %empty_unpack_b =
      transform.structured.pack_transpose %pack_producer_b with_compute_op(%packed_a)
      outer_perm = [0, 1, 3, 2] inner_perm = [1, 0] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Transpose C matrix from [M N m n m0 n0] to [M N n m m0 n0]
    %unpack = transform.get_consumers_of_result %packed_b[0]
      : (!transform.any_op) -> (!transform.any_op)
    %packed_c, %pack_c, %unpack_c =
      transform.structured.pack_transpose %unpack with_compute_op(%packed_b)
      outer_perm = [0, 1, 3, 2] : (!transform.any_op, !transform.any_op)
      -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    transform.print %variant_op : !transform.any_op

    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()

    // Bufferize and drop HAL decriptor from memref ops.
    %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> !transform.any_op
    transform.print %variant_op_3 : !transform.any_op
  }
}
