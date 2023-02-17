// Repro:
// export IREE_SAMPLES_DIR=/usr/local/google/home/ntv/github; \
// iree-opt tests/e2e/models/bert_encoder_unrolled_fake_weights.mlir --iree-mhlo-to-mhlo-preprocessing --iree-mhlo-to-linalg-on-tensors | \
// iree-opt --pass-pipeline="builtin.module(func.func(iree-transform-dialect-interpreter{transform-file-name=${IREE_SAMPLES_DIR}/iree-samples/transform_dialect/bert_preprocessing_spec.mlir}))" | \
// iree-opt --transform-dialect-drop-schedule | \
// iree-opt --iree-import-public \
//          --iree-mhlo-input-transformation-pipeline \
//          --iree-tosa-input-transformation-pipeline \
//          --iree-abi-transformation-pipeline \
//          --iree-flow-transformation-pipeline \
//          --iree-stream-transformation-pipeline

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %matmul_384_128 = transform.structured.match ops{["linalg.matmul"]} filter_result_type = tensor<384x128xf32> in %module_op 
    : (!pdl.operation) -> (!pdl.operation)
  // TODO: pack_greedily with next_multiple_of
  transform.structured.pack_greedily %matmul_384_128
      gemm_packed_sizes = [16, 16, 512] gemm_inner_dims_order = [0, 1, 2]
    : (!pdl.operation) -> !transform.op<"linalg.generic">

  %matmul_384_2 = transform.structured.match ops{["linalg.matmul"]} filter_result_type = tensor<384x2xf32> in %module_op 
    : (!pdl.operation) -> (!pdl.operation)
  // TODO: pack_greedily with next_multiple_of
  transform.structured.pack_greedily %matmul_384_2
      gemm_packed_sizes = [16, 16, 512] gemm_inner_dims_order = [0, 1, 2]
    : (!pdl.operation) -> !transform.op<"linalg.generic">

  %matmul_384_512 = transform.structured.match ops{["linalg.matmul"]} filter_result_type = tensor<384x512xf32> in %module_op 
    : (!pdl.operation) -> (!pdl.operation)
  // TODO: pack_greedily with next_multiple_of
  transform.structured.pack_greedily %matmul_384_512
      gemm_packed_sizes = [16, 16, 384] gemm_inner_dims_order = [0, 1, 2]
    : (!pdl.operation) -> !transform.op<"linalg.generic">

  %batch_matmul_4_384_384 = transform.structured.match ops{["linalg.batch_matmul"]} filter_result_type = tensor<4x384x384xf32> in %module_op 
    : (!pdl.operation) -> (!pdl.operation)
  // TODO: pack_greedily with next_multiple_of
  transform.structured.pack_greedily %batch_matmul_4_384_384
      gemm_packed_sizes = [16, 16, 32] gemm_inner_dims_order = [0, 1, 2]
    : (!pdl.operation) -> !transform.op<"linalg.generic">

  %batch_matmul_4_384_32 = transform.structured.match ops{["linalg.batch_matmul"]} filter_result_type = tensor<4x384x32xf32> in %module_op 
    : (!pdl.operation) -> (!pdl.operation)
  // TODO: pack_greedily with next_multiple_of
  transform.structured.pack_greedily %batch_matmul_4_384_32
      gemm_packed_sizes = [16, 16, 384] gemm_inner_dims_order = [0, 1, 2]
    : (!pdl.operation) -> !transform.op<"linalg.generic">

  // This is a rewrite of tensor.pack/tensor.unpack to linalg_ext.pack/linalg_ext.unpack
  // that IREE currently understands.
  // TODO: Remove once IREE adopts tensor.pack/unpack.
  // TODO: Unfortunately, this does not go through and hangs in iree-compile so
  // we need to fallback to other lowering to linalg.fill/linalg.transpose/etc below.
  //
  // %func = transform.structured.match ops{["func.func"]} in %module_op
  //   : (!pdl.operation) -> (!pdl.operation)
  // transform.iree.apply_patterns %func { rewrite_pack_ops }

  // IREE does not understand tensor.pack/unpack yet, so we have to lower them
  // explicitly to a form IREE understands.
  // This is only required to generate the PTX.
  //
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
}
