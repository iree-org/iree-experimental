
// Instructions
// ============
//
// This example demonstrates:
//   1. The creation of a split-K strategy, for unaligned cases, using forall at
//      the graph level.
//   2. Fusion of the newly created fill in the scf.forall at the graph level.
//   3. Mechanical conversion of the scf.forall to a dispatch region.
//   4. Further composition with codegen within the dispatch region.
//
// The following command line can be used to only apply transforms at the graph
// level in a preprocessing fashion:
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
//   sed "s/\${M}/123/g" | sed "s/\${N}/456/g" | sed "s/\${K}/51234/g" | \
//   sed "s/private @matmul_static(/@matmul_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce | \
//   iree-opt --pass-pipeline="builtin.module(func.func(iree-transform-dialect-interpreter{transform-file-name=${IREE_SAMPLES_DIR}/transform_dialect/graph/example_split_k/matmul_split_k_graph_spec.mlir}))"
// ```
//
// The following command line can be used to:
//   1. Apply transforms at the graph level to perform unaligned split-K with an
//      scf.forall + fusion.
//   2. Convert only the scf.forall to a dispatch region (i.e. custom dispatch
//      region formation).
//   3. Apply default IREE dispatch region formation + codegen to the rest of
//      the graph (i.e. the reduction part).
//   4. Apply custom codegen to the dispatch region using another transform file.
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
//   sed "s/\${M}/123/g" | sed "s/\${N}/456/g" | sed "s/\${K}/51234/g" | \
//   sed "s/private @matmul_static(/@matmul_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce | \
//   iree-opt  --iree-hal-target-backends=cuda \
//             --iree-abi-transformation-pipeline \
//             --iree-flow-transformation-pipeline \
//             --iree-flow-dispatch-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/graph/example_split_k/matmul_split_k_graph_spec.mlir \
//             --iree-stream-transformation-pipeline \
//             --iree-hal-configuration-pipeline | \
//   iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
//      --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/graph/example_split_k/matmul_split_k_codegen_spec.mlir \
//      --iree-codegen-llvmgpu-enable-transform-dialect-jit=false
// ```
//
// When `iree.populate_workgroup_count_region_using_num_threads_slice` and 
// 

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %module_op 
    : (!pdl.operation) -> (!pdl.operation)

  // Tile reduction to forall, which also pads on the fly.
  // =====================================================
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %module_op
    : (!pdl.operation) -> !pdl.operation
  %forall, %fill, %meatier_matmul, %tail_reduction = transform.structured.tile_reduction_using_forall 
    %matmul by num_threads = [0, 0, 77], tile_sizes = [], mapping = [#gpu.block<z>]
  %fill_l1 = transform.structured.fuse_into_containing_op %fill into %forall

  %region_op = transform.iree.wrap_in_dispatch_region %forall { generateWorkload = true }
  transform.iree.region_to_workgroups %region_op
}
