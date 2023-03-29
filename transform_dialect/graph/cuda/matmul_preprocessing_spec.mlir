
// Instructions
// ============
//
// Note: this depends on the integrate: https://github.com/openxla/iree/pull/12822
//
// Apply transforms as a preprocessing at the graph level and drop the schedule.
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
//   sed "s/\${M}/999/g" | sed "s/\${N}/3999/g" | sed "s/\${K}/1999/g" | \
//   sed "s/private @matmul_static(/@matmul_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce | \
//   iree-opt --pass-pipeline="builtin.module(func.func(iree-transform-dialect-interpreter{transform-file-name=${IREE_SAMPLES_DIR}/transform_dialect/graph/cuda/matmul_preprocessing_spec.mlir}))" | \
//   iree-compile - --iree-hal-target-backends=cuda --iree-hal-benchmark-dispatch-repeat-count=5 | \
//   nsys profile --stats=true  iree-run-module --function=matmul_static --device=cuda --input=999x1999xf32=1 --input=1999x3999xf32=1 --input=999x3999xf32=1
//
// # Alternatively:
// # nsys nvprof --print-gpu-trace  iree-run-module --function=forward --device=cuda --input=999x1999xf32=1 --input=1999x3999xf32=1 --input=999x3999xf32=1
// ```
//
// To run e2e on a remote machine (${A100_MACHINE_IP}) with an A100 GPU:
// ```
//   # Do this only once:
//   # scp ${IREE_DIR}/build/tools/iree-run-module ${USER}@${A100_MACHINE_IP}:~/;
//
//   export IREE_DIR=${HOME}/github/iree; 
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; 
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir | \
//   sed "s/\${M}/999/g" | sed "s/\${K}/1999/g" | sed "s/\${N}/3999/g" | \
//   sed "s/private @fill_matmul_static(/@fill_matmul_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   iree-opt --pass-pipeline="builtin.module(func.func(iree-transform-dialect-interpreter{transform-file-name=${IREE_SAMPLES_DIR}/transform_dialect/graph/cuda/matmul_preprocessing_spec.mlir}))" | \
//   ${IREE_DIR}/build/tools/iree-compile - \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-hal-benchmark-dispatch-repeat-count=5 \
//     -o /tmp/foo.vmfb; \
//   scp /tmp/foo.vmfb ${USER}@${A100_MACHINE_IP}:~/ > /dev/null; \
//   ssh ${USER}@${A100_MACHINE_IP} "/usr/local/cuda/bin/nsys profile --stats=true ~/iree-run-module --function=fill_matmul_static --device=cuda --module=foo.vmfb --input=999x1999xf32=1 --input=1999x3999xf32=1 --input=999x3999xf32=1 2>&1" | \
//   grep fill_matmul_static_dispatch | awk '{print $6}'


transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %module_op 
    : (!pdl.operation) -> (!pdl.operation)

  // Step 1. Generic packing with reduction padding to the next multiple of 64.
  // ==========================================================================
  %generic = transform.structured.match ops{["linalg.matmul"]} in %module_op
    : (!pdl.operation) -> !pdl.operation
  transform.structured.pack_greedily %generic
      matmul_packed_sizes = [0, 0, 0] 
      matmul_padded_sizes_next_multiple_of = [128, 128, 128] 
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

  // TODO: need more stuff to cleanup the rank-reducing-reshape + transpose + 
  // rank-reducing-insert/extract chains and trn those into exactly a pad.
  transform.iree.apply_patterns %func { canonicalization, cse } : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func { rank_reducing_linalg } : (!pdl.operation) -> ()
  transform.iree.apply_patterns %func { canonicalization, cse } : (!pdl.operation) -> ()
}
