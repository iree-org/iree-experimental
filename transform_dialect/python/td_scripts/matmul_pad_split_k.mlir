// export IREE_DIR=${HOME}/github/iree; \
// export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
// cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
// sed "s/\${M}/123/g" | sed "s/\${N}/456/g" | sed "s/\${K}/12345/g" | \
// sed "s/private @fill_matmul_static(/@fill_matmul_static(/g" | \
// ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce | \
// iree-opt  --iree-hal-target-backends=cuda \
//           --iree-abi-transformation-pipeline \
//           --iree-flow-transformation-pipeline \
//           --iree-flow-dispatch-use-transform-dialect=./matmul_pad_split_k.mlir \
//           --iree-stream-transformation-pipeline \
//           --iree-hal-configuration-pipeline \
//           --iree-flow-enable-pad-handling | 
// iree-compile - \
//           --iree-flow-enable-pad-handling \
//           --iree-codegen-llvmgpu-enable-transform-dialect-pad-strategy \
//           --mlir-disable-threading \
//           --debug-only=iree-transform-strategy-builder

transform.sequence failures(propagate) {
^bb1(%module_op: !transform.any_op):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %module_op
    : (!transform.any_op) -> !transform.any_op

  %matmul_padded = transform.structured.pad %matmul {
    padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
    padding_dimensions=[0, 1, 2],
    pad_to_multiple_of=[32, 32, 1728], // 1728 = 108 * 16 on ampere
    pack_paddings=[0, 0, 0]
  } : (!transform.any_op) -> !transform.any_op

  // Without insert_split_dimension = 2, tileAnedUnrolConv(!!) crashes in the 
  // IREE default codegen pipeline.
  %1:4 = transform.structured.split_reduction %matmul_padded 
    { split_factor = 108, insert_split_dimension = 2 } 
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op, !transform.any_op)
  // Without iterator_interchange = [2, 0, 1, 3], the IREE default codegen 
  // pipeline crashes with:
  //   513216 bytes of shared memory; exceeded the limit of 166912 bytes
  transform.structured.interchange %1#2 iterator_interchange = [2, 0, 1, 3]
    : (!transform.any_op) -> !transform.any_op
  // With the above 2, the IREE default codegen pipeline passes .. 

  // Optional cleanups.
  %func = transform.structured.match ops{["func.func"]} in %module_op 
    : (!transform.any_op) -> (!transform.any_op)
  transform.apply_patterns to %func {
    transform.apply_patterns.tensor.drop_redundant_insert_slice_rank_expansion
  } : !transform.any_op
  transform.iree.apply_patterns %func { canonicalization, cse } : (!transform.any_op) -> ()

  // Warning: if passed to iree-compile, this will pollute the vmfb and weird 
  // error messages will print.
  // transform.print %func {name = "after preprocessing"} : !transform.any_op
}
