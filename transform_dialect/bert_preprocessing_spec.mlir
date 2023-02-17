// Instructions; TL;DR
// ===================
// ```
//   export IREE_DIR=/usr/local/google/home/ntv/github/iree; \
//   export IREE_SAMPLES_DIR=/usr/local/google/home/ntv/github/iree-samples; \
//   iree-opt ${IREE_DIR}/tests/e2e/models/bert_encoder_unrolled_fake_weights.mlir --iree-mhlo-to-mhlo-preprocessing --iree-mhlo-to-linalg-on-tensors | \
//   iree-opt --pass-pipeline="builtin.module(func.func(iree-transform-dialect-interpreter{transform-file-name=${IREE_SAMPLES_DIR}/transform_dialect/graph/cuda/bert_preprocessing_spec.mlir}))" | \
//   iree-compile - --iree-hal-target-backends=cuda | \
//   nsys profile --stats=true  iree-run-module --function=serving_default --device=cuda
// ```

// Instructions
// ============
// 1. Apply transforms as a preprocessing at the graph level and drop the schedule.
//    Note: iree-opt --pass-pipeline="builtin.module(func.func(iree-transform-dialect-interpreter....."
//          could also be replaced by:
//          mlir-opt --pass-pipeline="builtin.module(test-transform-dialect-interpreter{transform-file-name=...})"
//    When/if IREE accepts a transform interpreter in the preprocessing step, the commands can get significantly simpler.
// ```
//   export IREE_DIR=/usr/local/google/home/ntv/github/iree; \
//   export IREE_SAMPLES_DIR=/usr/local/google/home/ntv/github/iree-samples; \
//   iree-opt ${IREE_DIR}/tests/e2e/models/bert_encoder_unrolled_fake_weights.mlir --iree-mhlo-to-mhlo-preprocessing --iree-mhlo-to-linalg-on-tensors | \
//   iree-opt --pass-pipeline="builtin.module(func.func(iree-transform-dialect-interpreter{transform-file-name=${IREE_SAMPLES_DIR}/transform_dialect/graph/cuda/bert_preprocessing_spec.mlir}))"
// ```
//
// 2. Optionally, pipe through the following command to get to a custom TD codegen file.
// ```
//   iree-opt --iree-import-public \
//            --iree-mhlo-input-transformation-pipeline \
//            --iree-tosa-input-transformation-pipeline \
//            --iree-abi-transformation-pipeline \
//            --iree-flow-transformation-pipeline \
//            --iree-stream-transformation-pipeline | \
//  iree-opt <incantations to custom TD codegen file>
// ```
//
// 3. Pipe through iree-compile to get PTX with IREE's default C++ codegen.
//    Note: --iree-hal-benchmark-dispatch-repeat-count=5 because the first iteration
//    is very expensive due to host-device memory transfers.
// ```
//   # If using the faster `nsys profile --stats=true`, the first reported 
//   # iteration is very expensive due to host-device memory transfers
//   iree-compile - --iree-hal-target-backends=cuda --iree-hal-benchmark-dispatch-repeat-count=5
//
//   # if using the deeper `ncu -f --set full -o profile`, just benchmark 1 iteration.
//   iree-compile - --iree-hal-target-backends=cuda
// ```
//
// 4. Run and measure with nsys (faster) / ncu (slower) (requires sudo):
// ```
//   nsys profile --stats=true iree-run-module --function=serving_default --device=cuda
//   sudo ${NCU_PATH}/ncu -f --set full -o profile ${IREE_RUN_MODULE_PATH}/iree-run-module --function=serving_default --device=cuda
// ```
//
// The above generates something like:
// ```
//  Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)     GridXYZ         BlockXYZ                             Name                         
// --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  --------------  --------------  -----------------------------------------------------
//     45.4      114,913,101        168    684,006.6    678,437.0    658,917    713,896     20,250.3     1    1    8    32    8    1  _serving_default_dispatch_14_generic_24x8x16x16x512  
//     27.3       69,062,480        121    570,764.3    567,904.0    549,215    594,977     16,799.9     1    1   32    32    8    1  _serving_default_dispatch_6_generic_24x32x16x16x384  
//     21.2       53,737,526         24  2,239,063.6  2,207,563.0  2,163,321  2,354,979     74,657.1     1    1    2    32    8    1  _serving_default_dispatch_35_generic_4x24x2x16x16x384
//      3.2        8,222,893         24    342,620.5    340,227.0    329,267    355,956     10,211.7     1    1   24    32    8    1  _serving_default_dispatch_29_generic_4x24x24x16x16x32
// ```
//
// 5. Run and measure with `iree-benchmark-module` (TODO commands for tracy etc)
// ```
//   iree-benchmark-module --entry_function=serving_default --device=cuda
// ```


// This script currently does whatever is needed to connect e2e to IREE; a bunch
// of improvements are still needed:
//   1. make pack_greedily more automatic with a `next_multiple_of` option.
//   2. avoid lowering of tensor.pack/unpack once things just work with these
//      abstractions in IREE.
//   3. hook up pack/unpack propagation patterns upstream and just call here.
//
// When these 3 items are done, the script below should reduce to ~10 lines.
transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %matmul_384_128 = transform.structured.match ops{["linalg.matmul"]} filter_result_type = tensor<384x128xf32> in %module_op 
    : (!pdl.operation) -> (!pdl.operation)
  // TODO: pack_greedily with next_multiple_of to generalize across reduction sizes that are not captured by filter_result_type
  // i.e. we should have something resembling gemm_packed_sizes = [16, 16, next_multiple_of(16)] for all linalg.matmul
  transform.structured.pack_greedily %matmul_384_128
      gemm_packed_sizes = [16, 16, 512] gemm_inner_dims_order = [0, 1, 2]
    : (!pdl.operation) -> !transform.op<"linalg.generic">

  %matmul_384_2 = transform.structured.match ops{["linalg.matmul"]} filter_result_type = tensor<384x2xf32> in %module_op 
    : (!pdl.operation) -> (!pdl.operation)
  // TODO: pack_greedily with next_multiple_of to generalize across reduction sizes that are not captured by filter_result_type
  // i.e. we should have something resembling gemm_packed_sizes = [16, 16, next_multiple_of(16)] for all linalg.matmul
  transform.structured.pack_greedily %matmul_384_2
      gemm_packed_sizes = [16, 16, 512] gemm_inner_dims_order = [0, 1, 2]
    : (!pdl.operation) -> !transform.op<"linalg.generic">

  %matmul_384_512 = transform.structured.match ops{["linalg.matmul"]} filter_result_type = tensor<384x512xf32> in %module_op 
    : (!pdl.operation) -> (!pdl.operation)
  // TODO: pack_greedily with next_multiple_of to generalize across reduction sizes that are not captured by filter_result_type
  // i.e. we should have something resembling gemm_packed_sizes = [16, 16, next_multiple_of(16)] for all linalg.matmul
  transform.structured.pack_greedily %matmul_384_512
      gemm_packed_sizes = [16, 16, 384] gemm_inner_dims_order = [0, 1, 2]
    : (!pdl.operation) -> !transform.op<"linalg.generic">

  %batch_matmul_4_384_384 = transform.structured.match ops{["linalg.batch_matmul"]} filter_result_type = tensor<4x384x384xf32> in %module_op 
    : (!pdl.operation) -> (!pdl.operation)
  // TODO: pack_greedily with next_multiple_of to generalize across reduction sizes that are not captured by filter_result_type
  // i.e. we should have something resembling gemm_packed_sizes = [16, 16, next_multiple_of(16)] for all linalg.batch_matmul
  transform.structured.pack_greedily %batch_matmul_4_384_384
      gemm_packed_sizes = [16, 16, 32] gemm_inner_dims_order = [0, 1, 2]
    : (!pdl.operation) -> !transform.op<"linalg.generic">

  %batch_matmul_4_384_32 = transform.structured.match ops{["linalg.batch_matmul"]} filter_result_type = tensor<4x384x32xf32> in %module_op 
    : (!pdl.operation) -> (!pdl.operation)
  // TODO: pack_greedily with next_multiple_of to generalize across reduction sizes that are not captured by filter_result_type
  // i.e. we should have something resembling gemm_packed_sizes = [16, 16, next_multiple_of(16)] for all linalg.batch_matmul
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
