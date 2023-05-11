// Instructions; TL;DR
// ===================
//
// Note: this is currently dependent on WIP in the branch:
//   https://github.com/nicolasvasilache/iree/tree/matmul-unaligned
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/copy.mlir |\
//   sed "s/\${M}/5/g" |\
//   sed "s/private @copy_1d_static(/@copy_1d_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   ${IREE_DIR}/build/tools/iree-opt \
//     --iree-hal-target-backends=cuda \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_DIR}/build/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
//      --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/copy_1d_codegen_spec.mlir \
//      --iree-codegen-llvmgpu-enable-transform-dialect-jit=false
// ```

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %copy = transform.structured.match ops{["linalg.copy"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  %forall_l1, %copy_l1 =
    transform.structured.tile_to_forall_op %copy num_threads [2]
      ( mapping = [#gpu.block<x>] )
  transform.iree.populate_workgroup_count_region_using_num_threads_slice
    %forall_l1 : (!pdl.operation) -> ()

  // Late canonicalizations and cleanups.
  transform.iree.apply_patterns %variant_op
    {canonicalization, cse, licm, tiling_canonicalization}
    : (!pdl.operation) -> ()
}
