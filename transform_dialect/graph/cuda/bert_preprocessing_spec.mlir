// Instructions; TL;DR
// ===================
//
// Note: depends onf https://github.com/nicolasvasilache/iree/tree/fold-tensor-subset
// which is currently blocked by IREE / LLVM integrate.
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
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
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   iree-opt ${IREE_DIR}/tests/e2e/models/bert_encoder_unrolled_fake_weights.mlir --iree-mhlo-to-mhlo-preprocessing --iree-mhlo-to-linalg-on-tensors | \
//   iree-opt --pass-pipeline="builtin.module(func.func(iree-transform-dialect-interpreter{transform-file-name=${IREE_SAMPLES_DIR}/transform_dialect/graph/cuda/bert_preprocessing_spec.mlir}))"
// ```
//
// 2. Optionally, pipe through the following command to get to a custom TD codegen file.
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   iree-opt ${IREE_DIR}/tests/e2e/models/bert_encoder_unrolled_fake_weights.mlir --iree-mhlo-to-mhlo-preprocessing --iree-mhlo-to-linalg-on-tensors | \
//   iree-opt --pass-pipeline="builtin.module(func.func(iree-transform-dialect-interpreter{transform-file-name=${IREE_SAMPLES_DIR}/transform_dialect/graph/cuda/bert_preprocessing_spec.mlir}))" | \
//   iree-opt - --iree-import-public \
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


// Step 1. Generic packing with reduction padding to the next multiple of 16.
// ==========================================================================
transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %module_op 
    : (!pdl.operation) -> (!pdl.operation)
  transform.structured.pack_greedily %matmul
      matmul_packed_sizes = [0, 0, 0] 
      matmul_padded_sizes_next_multiple_of = [16, 16, 64] 
      matmul_inner_dims_order = [0, 1, 2]
    : (!pdl.operation) -> !transform.op<"linalg.generic">

  %batch_matmul = transform.structured.match ops{["linalg.batch_matmul"]} in %module_op 
    : (!pdl.operation) -> (!pdl.operation)
  transform.structured.pack_greedily %batch_matmul
      matmul_packed_sizes = [0, 0, 0] 
      matmul_padded_sizes_next_multiple_of = [16, 16, 64]
      matmul_inner_dims_order = [0, 1, 2]
    : (!pdl.operation) -> !transform.op<"linalg.generic">

  // Step 2. Pack / unpack propagation (TODO: activate when it works).
  // =================================================================
  // Pack / unpack propagation does not yet work properly and produces invalid IR.
  // E.g.:
  // <stdin>:17141:13: error: 'tensor.pack' op invalid tile factor provided. 
  // Only full tiles are supported when padding_value is not set
  // %6602 = linalg.generic {indexing_maps = [#map2, #map3], 
  //                         iterator_types = ["parallel", "parallel"]} 
  //         ins(%1114 : tensor<2xf32>) outs(%6601 : tensor<384x2xf32>) ...
  //
  // %func = transform.structured.match ops{["func.func"]} in %module_op 
  //   : (!pdl.operation) -> (!pdl.operation)
  // transform.iree.apply_patterns %func { bubble_pack_un_pack }
  //   : (!pdl.operation) -> ()

  // Step 3. Special pack / unpack lowering (TODO: drop when possible).
  // ==================================================================
  // IREE fails lowering of tensor.pack/unpack ops with:
  // <stdin>:23295:20: error: 'tensor.extract_slice' op unhandled operation when 
  // converting to destination passing style
  // %unpack_3675 = tensor.unpack %7830 inner_dims_pos = [0, 1] 
  //                inner_tiles = [384, 16] into %7826 
  //                : tensor<1x1x384x16xf32> -> tensor<384x2xf32>
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

  // Step 4. Various patterns (TODO: drop when possible).
  // ====================================================
  %func = transform.structured.match ops{["func.func"]} in %module_op 
    : (!pdl.operation) -> (!pdl.operation)
  transform.iree.apply_patterns %func { rank_reducing_linalg_via_reshapes } : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func { canonicalization, cse } : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func { linalg_elementwise_greedy_fusion } : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func { bubble_expand } : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func { bubble_collapse } : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func { canonicalization, cse } : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func { fold_reassociative_reshape } : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func { linalg_elementwise_greedy_fusion } : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func { fold_tensor_empty_extract } : (!pdl.operation) -> ()
}

