
// Instructions
// ============
//
// Apply transforms as a preprocessing at the graph level.
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
//   sed "s/\${M}/123/g" | sed "s/\${N}/456/g" | sed "s/\${K}/51234/g" | \
//   sed "s/private @matmul_static(/@matmul_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce | \
//   iree-opt --pass-pipeline="builtin.module(func.func(iree-transform-dialect-interpreter{transform-file-name=${IREE_SAMPLES_DIR}/transform_dialect/graph/cuda/matmul_large_K_preprocessing_spec.mlir}))" | \
//   iree-compile - --iree-hal-target-backends=cuda --iree-hal-benchmark-dispatch-repeat-count=5 | \
//   nsys profile --stats=true  iree-run-module --function=matmul_static --device=cuda --input=123x51234xf32=1 --input=51234x456xf32=1 --input=123x456xf32=1
//
// # Alternatively:
// # nsys nvprof --print-gpu-trace  iree-run-module --function=forward --device=cuda --input=123x51234xf32=1 --input=51234x456xf32=1 --input=123x456xf32=1
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
//   sed "s/\${M}/123/g" | sed "s/\${K}/51234/g" | sed "s/\${N}/456/g" | \
//   sed "s/private @fill_matmul_static(/@fill_matmul_static(/g" | \
//   ${LLVM_BUILD_DIR}/bin/mlir-opt -symbol-dce |
//   iree-opt --pass-pipeline="builtin.module(func.func(iree-transform-dialect-interpreter{transform-file-name=${IREE_SAMPLES_DIR}/transform_dialect/graph/cuda/matmul_large_K_preprocessing_spec.mlir}))" | \
//   ${IREE_DIR}/build/tools/iree-compile - \
//     --iree-hal-target-backends=cuda --iree-hal-cuda-llvm-target-arch=sm_80 \
//     --iree-flow-fuse-multi-use \
//     --iree-hal-benchmark-dispatch-repeat-count=5 \
//     -o /tmp/foo.vmfb; \
//   scp /tmp/foo.vmfb ${USER}@${A100_MACHINE_IP}:~/ > /dev/null; \
//   ssh ${USER}@${A100_MACHINE_IP} "/usr/local/cuda/bin/nsys profile --stats=true ~/iree-run-module --function=fill_matmul_static --device=cuda --module=foo.vmfb --input=123x51234xf32=1 --input=51234x456xf32=1 --input=123x456xf32=1 2>&1" | \
//   grep fill_matmul_static_dispatch | awk '{print $6}'
//
// Extra iree-compile commands: (mma-sync actually reduces perf here)
//   -iree-codegen-llvmgpu-use-mma-sync \
// ```

transform.sequence failures(propagate) {
^bb1(%module_op: !pdl.operation):
  %func = transform.structured.match ops{["func.func"]} in %module_op 
    : (!pdl.operation) -> (!pdl.operation)

  // Step 1. Generic packing with reduction padding to next multiple of 3456 (i.e. 108 * 32).
  // ========================================================================================
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %module_op
    : (!pdl.operation) -> !pdl.operation
  %packed_matmul = transform.structured.pack_greedily %matmul 
      matmul_packed_sizes = [0, 0, 0] 
      matmul_padded_sizes_next_multiple_of = [128, 256, 3456] // 3456 = 108 * 32
      // We want [0, 2, 1] to get back to a mn, mk, kn ordering.
      // Otherwise we'd get mn, mk, nk.
      matmul_inner_dims_order = [0, 2, 1]
    : (!pdl.operation) -> !transform.op<"linalg.generic">

  // Need to apply rank-reducing patterns so that split_reduction only sees a single
  // reduction dimension. Otherwise, split_reduction has multiple options and chokes atm.
  // The upstream fix is simple, add a parameter to specify which of the reductions
  // should be split and use the most minor one as the default.
  transform.iree.apply_patterns %module_op {rank_reducing_linalg} : (!pdl.operation) -> ()

  %packed_matmul_cast = 
    transform.cast %packed_matmul : !transform.op<"linalg.generic"> to !pdl.operation
  %1:4 = transform.structured.split_reduction %packed_matmul_cast 
    { split_factor = 54, insert_split_dimension = 0 }

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

  // Without generalize, the linalg.transpose named op triggers bad fusion
  // heuristics that result in very expensive tranpose kernels.
  // With transform.structured.generalize, the overhead is more manageable.
  %generic = transform.structured.match interface{LinalgOp} in %module_op
    : (!pdl.operation) -> !pdl.operation
  transform.structured.generalize %generic
  
  transform.iree.apply_patterns %func { canonicalization, cse } : (!pdl.operation) -> ()
}
