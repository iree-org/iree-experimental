// Instructions; TL;DR
// ===================
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/pack.mlir |\
//   sed "s/\${N}/999/g" | sed "s/\${M}/1999/g" |\
//   sed "s/\${Npad}/1024/g" | sed "s/\${Mpad}/2048/g" |\
//   ${IREE_DIR}/build/tools/iree-opt \
//     --iree-hal-target-backends=cuda \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_DIR}/build/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
//      --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/pad_via_pack_pipeline_spec.mlir \
//      --iree-codegen-llvmgpu-enable-transform-dialect-jit=false
// ```
//
// To produce PTX:
// ```
//   export IREE_DIR=${HOME}/github/iree;
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples;
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/pack.mlir | \
//   sed "s/\${N}/999/g" | sed "s/\${M}/1999/g" |\
//   sed "s/\${Npad}/1024/g" | sed "s/\${Mpad}/2048/g" |\
//   ${IREE_DIR}/build/tools/iree-compile - \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/pad_via_pack_pipeline_spec.mlir \
//     --iree-codegen-llvmgpu-enable-transform-dialect-jit=false
// ```

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  transform.print %variant_op : !pdl.operation

  // Step 1. Convert pack/unpack to pad/extract_slice.
  %pack = transform.structured.match ops{["tensor.pack"]} in %variant_op
    : (!pdl.operation) -> !transform.op<"tensor.pack">
  transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">)
    -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">,
        !transform.op<"linalg.transpose">)
  %unpack = transform.structured.match ops{["tensor.unpack"]} in %variant_op
    : (!pdl.operation) -> !transform.op<"tensor.unpack">
  transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">)
    -> (!transform.op<"tensor.empty">,
        !transform.op<"linalg.transpose">,
        !transform.op<"tensor.collapse_shape">,
        !transform.op<"tensor.extract_slice">)

  transform.print %variant_op {name = "after conversion to pad"} : !pdl.operation

  // Step 2. Tile and distribute.
  %pad = transform.structured.match ops{["tensor.pad"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  %forall_pad, %tiled_pad = transform.structured.tile_to_forall_op %pad
    tile_sizes [8, 32] ( mapping = [#gpu.block<y>, #gpu.block<x>] )
  transform.iree.populate_workgroup_count_region_using_num_threads_slice
    %forall_pad : (!pdl.operation) -> ()

  // There might be an additional level of tiling needed, e.g. first-level
  // tiling has tile sizes [8, 128], second-level has tile sizes [1, 4]. In
  // that case, we should be prepared to vectorize `scf.if`, because we won't
  // be able to take the assumed branch as we do now for the first level of
  // tiling.
  %if = transform.structured.match ops{["scf.if"]} in %forall_pad
    : (!pdl.operation) -> !transform.any_op
  transform.scf.take_assumed_branch %if take_else_branch
    : (!transform.any_op) -> ()

  transform.iree.apply_patterns %variant_op
    {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()

  transform.print %variant_op {name = "after tiling"} : !pdl.operation

  // Step 3. Vectorize pad.
  %pad_inside = transform.structured.match ops{["tensor.pad"]} in %forall_pad
    : (!pdl.operation) -> !pdl.operation

  transform.structured.masked_vectorize %pad_inside vector_sizes [8, 32]

  transform.iree.apply_patterns %variant_op
   {canonicalization, cse, licm, tiling_canonicalization} : (!pdl.operation) -> ()

  transform.print %variant_op {name = "after masked vectorization"} : !pdl.operation

  // Step 4. Rank-reduce and vectorize.
  // ==================================
  // Lower the masks to allow canonicalizations to kick in.
  %func_v = transform.structured.match ops{["func.func"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  %func_v_2 = transform.vector.lower_masked_transfers %func_v
    : (!pdl.operation) -> !pdl.operation
  transform.iree.apply_patterns %func_v_2
    { rank_reducing_linalg, rank_reducing_vector } : (!pdl.operation) -> ()
  %func_v_3 = transform.structured.vectorize %func_v_2

  transform.print %func_v_3 {name = "after vectorization"} : !pdl.operation

  // Step 5. Bufferize and drop HAL descriptor from memref ops.
  // ==========================================================
  // Pre-bufferization canonicalizations and cleanups help avoid extra copies.
  transform.iree.apply_patterns %func_v_3 {canonicalization, cse, licm}
    : (!pdl.operation) -> ()
  transform.iree.eliminate_empty_tensors %func_v_3 : (!pdl.operation) -> ()
  %variant_op_bufferized = transform.iree.bufferize { target_gpu } %variant_op
   : (!pdl.operation) -> (!pdl.operation)

  %func_m = transform.structured.match ops{["func.func"]}
   in %variant_op_bufferized : (!pdl.operation) -> !pdl.operation
  transform.iree.erase_hal_descriptor_type_from_memref %func_m
    : (!pdl.operation) -> ()

  // Step 4. Post-bufferization mapping workgroup.
  // =============================================
  transform.iree.forall_to_workgroup %func_m: (!pdl.operation) -> ()

  // Step 4. Map to wmm ops.
  // =======================
  // This must occur after bufferization because of the fancy CUDA types.
  transform.iree.vector.vector_to_mma_conversion %func_m { use_wmma }
    : (!pdl.operation) -> ()

  // Late canonicalizations and cleanups.
  transform.iree.apply_patterns %func_m
    {canonicalization, cse, licm, tiling_canonicalization}
      : (!pdl.operation) -> ()
}
