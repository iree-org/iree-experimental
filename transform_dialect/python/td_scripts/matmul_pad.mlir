
// export IREE_DIR=${HOME}/github/iree; \
// export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
// cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
// sed "s/\${M}/123/g" | sed "s/\${N}/456/g" | sed "s/\${K}/12345/g" | \
// sed "s/private @fill_matmul_static(/@fill_matmul_static(/g" | \
// ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce | \
// iree-opt  --iree-hal-target-backends=cuda \
//           --iree-abi-transformation-pipeline \
//           --iree-flow-transformation-pipeline \
//           --iree-flow-dispatch-use-transform-dialect=./matmul_pad.mlir \
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
  // TODO: we need better filters here to build a simple heuristic:
  //   Case 1. only match unaligned matmuls and align them to the next 32x32x16.
  //   Case 2. match matmuls with large K (e.g. > 4K) that is not divisible by 
  //   1728 (108 * 16) on ampere and align them to the next 32x32x1728.
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %module_op
    : (!transform.any_op) -> !transform.any_op

  %matmul_padded = transform.structured.pad %matmul {
    padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
    padding_dimensions=[0, 1, 2],
    pad_to_multiple_of=[32, 32, 1728], // 1728 = 108 * 16 on ampere
    pack_paddings=[0, 0, 0]
  } : (!transform.any_op) -> !transform.any_op

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

