
// Instructions
// ============
//
// Note: this depends on not-yet-integrated commits: https://github.com/openxla/iree/pull/12754
//
// Apply transforms as a preprocessing at the graph level and drop the schedule.
//
// ```
//   iree-opt transform_dialect/graph/attention_layer_bs_3.elided_weights.mlir \
//     --iree-mhlo-to-mhlo-preprocessing --iree-mhlo-to-linalg-on-tensors | \
//   iree-opt --pass-pipeline="builtin.module(func.func(iree-transform-dialect-interpreter{transform-file-name=transform_dialect/graph/cuda/bugs/3.mlir}))" | \
//   iree-compile - --iree-hal-target-backends=cuda --iree-hal-benchmark-dispatch-repeat-count=5 | \
//   iree-run-module --function=forward --device=cuda --input="3x17x768xf32=1"
// ```

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %module_op 
    : (!pdl.operation) -> (!pdl.operation)

  // Step 1. Generic packing with reduction padding to the next multiple of 64.
  // ==========================================================================
  %generic = transform.structured.match ops{["linalg.matmul", "linalg.batch_matmul"]} in %module_op
    : (!pdl.operation) -> !pdl.operation
  transform.structured.pack_greedily %generic
      matmul_packed_sizes = [0, 0, 0] 
      matmul_padded_sizes_next_multiple_of = [16, 16, 16] 
      matmul_inner_dims_order = [0, 1, 2]
    : (!pdl.operation) -> !transform.op<"linalg.generic">

  // Step 2. Special pack / unpack lowering (TODO: drop when possible).
  // ==================================================================
  // IREE fails lowering of tensor.pack/unpack ops with:
  // <stdin>:23295:20: error: 'tensor.extract_slice' op unhandled operation when 
  // converting to destination passing style
  // %unpack_3675 = tensor.unpack %7830 inner_dims_pos = [0, 1] 
  //                inner_tiles = [384, 16] into %7826 
  //                : tensor<1x1x384x16xf32> -> tensor<384x2xf32>
  //
  // So instead we lower them ourselves.
  %pack = transform.structured.match ops{["tensor.pack"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.pack">
  transform.structured.lower_pack %pack : (!transform.op<"tensor.pack">) 
    -> (!transform.op<"tensor.pad">, !transform.op<"tensor.expand_shape">, !transform.op<"linalg.transpose">)
  %unpack = transform.structured.match ops{["tensor.unpack"]} in %module_op
    : (!pdl.operation) -> !transform.op<"tensor.unpack">
  transform.structured.lower_unpack %unpack : (!transform.op<"tensor.unpack">) 
    -> (!transform.op<"tensor.empty">, 
        !transform.op<"linalg.transpose">,
        !transform.op<"tensor.collapse_shape">,
        !transform.op<"tensor.extract_slice">)

  // This seems to trigger miscompilations in IREE and produces 19205 etc
  //      CHECK: 3x17x768xf32=[
  // CHECK-SAME: [769 769 769 
  // Disabling the following lines passes the test.
  %linalg = transform.structured.match interface{LinalgOp} in %module_op
    : (!pdl.operation) -> !pdl.operation
  transform.structured.generalize %linalg
}
