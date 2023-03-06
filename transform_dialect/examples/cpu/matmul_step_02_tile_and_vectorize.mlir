// Instructions; TL;DR
// ===================
//
// This script shows an example of:
//   - first level of 3-D tiling to sequential loops (note: tiling on K does not 
//     divide and introduces a `?`).
//   - second level 3-D tiling to sequential loops (note: tiling by 1 on K turns 
//     the `?` into a 1 which lets us perform vectorization).
//   - vectorization.
//   - late bufferization (i.e. move from SSA + subsets semantics to 
//     side-effects semantics).
//   - lowering of vector operations all the way to LLVM.
//   - 
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
//   sed "s/\${M}/1024/g" | sed "s/\${K}/12345/g" | sed "s/\${N}/4096/g" | \
//   ${IREE_DIR}/build/tools/iree-opt \
//     --iree-hal-target-backends=llvm-cpu \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_DIR}/build/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target)))' \
//      --iree-codegen-llvmcpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cpu/matmul_step_02_tile_and_vectorize.mlir
// ```
//
// To execute:
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
//   sed "s/\${M}/1024/g" | sed "s/\${K}/12345/g" | sed "s/\${N}/4096/g" | \
//   iree-compile - --iree-hal-target-backends=llvm-cpu \
//     --iree-codegen-llvmcpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cpu/matmul_step_02_tile_and_vectorize.mlir | \
//   iree-run-module --function=matmul_static \
//     --input="1024x12345xf32=1" \
//     --input="12345x4096xf32=1" \
//     --input="1024x4096xf32=0" | \
//   FileCheck ${IREE_SAMPLES_DIR}/transform_dialect/examples/cpu/matmul_step_02_tile_and_vectorize.mlir
// ```
//
// CHECK: EXEC @matmul_static
// CHECK: result[0]: hal.buffer_view
// CHECK: 1024x4096xf32=[12345 12345 12345
// etc
//
//
// If we only performed steps 1. + 2. + 3. we would expect IR resembling:
//
// CHECK-IR: scf.for {{.*}} -> (tensor<1024x4096xf32>) {
// CHECK-IR:   scf.for {{.*}} -> (tensor<1024x4096xf32>) {
// CHECK-IR:     tensor.extract_slice {{.*}} [32, 128] [1, 1] : tensor<1024x4096xf32> to tensor<32x128xf32>
// CHECK-IR:     scf.for {{.*}} -> (tensor<32x128xf32>) {
// CHECK-IR:       scf.for {{.*}} -> (tensor<32x128xf32>) {
// CHECK-IR:         scf.for {{.*}} -> (tensor<32x128xf32>) {
// CHECK-IR:           vector.transfer_read {{.*}} {in_bounds = [true, true]} : tensor<32x128xf32>, vector<16x8xf32>
// CHECK-IR:           scf.for {{.*}} -> (vector<16x8xf32>) {
// CHECK-IR:             vector.transfer_read {{.*}} {in_bounds = [true, true]} : tensor<1024x2048xf32>, vector<16x1xf32>
// CHECK-IR:             vector.transfer_read {{.*}} {in_bounds = [true, true]} : tensor<2048x4096xf32>, vector<1x8xf32>
// CHECK-IR:             vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} 
// CHECK-IR-SAME:          : vector<16x1xf32>, vector<1x8xf32> into vector<16x8xf32>
// CHECK-IR:             scf.yield {{.*}} : vector<16x8xf32>
// CHECK-IR:           }
// CHECK-IR:           vector.transfer_write {{.*}} {in_bounds = [true, true]} : vector<16x8xf32>, tensor<32x128xf32>
// CHECK-IR:           scf.yield {{.*}} : tensor<32x128xf32>
// CHECK-IR:         }
// CHECK-IR:         scf.yield {{.*}} : tensor<32x128xf32>
// CHECK-IR:       }
// CHECK-IR:       scf.yield {{.*}} : tensor<32x128xf32>
// CHECK-IR:     }
// CHECK-IR:     tensor.insert_slice {{.*}} into {{.*}} [32, 128] [1, 1] : tensor<32x128xf32> into tensor<1024x4096xf32>
// CHECK-IR:     scf.yield %inserted_slice : tensor<1024x4096xf32>
// CHECK-IR:   }
// CHECK-IR:   scf.yield {{.*}} : tensor<1024x4096xf32>

// Comment parts of the IR below starting from the bottom-up and rerun the command
// to see the effects of step 1. ; step 1. + step 2.; etc..
transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %original_matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  // IREE-specific connection to the runtime and threadpool, required to run e2e.
  // ============================================================================
  %forall, %matmul =
    transform.iree.tile_to_forall_and_workgroup_count_region %original_matmul 
      num_threads [1]
      // TODO: IREE needs own workgroup mapping attribute independent of GPU.
      ( mapping = [#gpu.block<x>] )
  // Post-tiling canonicalizations, in particular to ensure num_threads == 1 in 
  // the IR.
  transform.iree.apply_patterns %variant_op 
    {canonicalization, cse, licm, tiling_canonicalization}

  // Step 1. Tile to forall and sequential scf.for.
  // ======================================================
  %matmul_l1, %loops_l1:3 = transform.structured.tile_to_scf_for %matmul [32, 128, 512]
  
  // Step 2. Tile to forall and sequential scf.for.
  // ======================================================
  %matmul_l2, %loops_l2:3 = transform.structured.tile_to_scf_for %matmul_l1 [8, 16, 1]
  %generic_l2 = transform.structured.generalize %matmul_l2

  // Step 3. Vectorize.
  // ======================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  %func_2 = transform.structured.vectorize %func

  // Post-vectorization canonicalizations and hoistings to avoid roundtripping 
  // vectors in memory and prepare for bufferization.
  transform.iree.apply_patterns %variant_op {canonicalization, cse, licm }
  %func_3 = transform.structured.hoist_redundant_tensor_subsets %func_2
    : (!pdl.operation) -> !pdl.operation

  // IREE-specific bufferization.
  // ============================================================================
  // Pre-buferization canonicalizations and cleanups help avoid extra copies.
  transform.iree.apply_patterns %variant_op {canonicalization, cse, licm}
  %variant_op_2 = transform.iree.bufferize %variant_op

  // IREE-specific cleanup and connection to the runtime and threadpool, required
  // to run e2e.
  // ============================================================================
  %func_e = transform.structured.match ops{["func.func"]} in %variant_op_2
    : (!pdl.operation) -> !pdl.operation
  %func_e_2 = transform.iree.erase_hal_descriptor_type_from_memref %func_e
  %func_e_3 = transform.iree.forall_to_workgroup %func_e_2
  
  // Step 4. Late blanket/intrusive lowering of vector ops to vector abstractions
  // that are close to the LLVM level-of abstraction.
  // ============================================================================
  %func_e_4 = transform.vector.lower_vectors %func_e_3

  // TODO: maybe control transform.lower_to_llvm from here.

  // Late canonicalizations and cleanups.
  transform.iree.apply_patterns %variant_op_2 {canonicalization, cse, licm}
}
