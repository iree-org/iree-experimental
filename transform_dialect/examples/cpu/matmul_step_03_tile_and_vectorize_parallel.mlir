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
//
// We use parallel tasks of size 256x256 as specified in the first transformation 
// `tile_to_forall_and_workgroup_count_region`.
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
//      --iree-codegen-llvmcpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cpu/matmul_step_03_tile_and_vectorize_parallel.mlir
// ```
//
// To execute and time on e.g. 16 threads (time here is very crude):
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
//   sed "s/\${M}/1024/g" | sed "s/\${K}/12345/g" | sed "s/\${N}/4096/g" | \
//   iree-compile - --iree-hal-target-backends=llvm-cpu \
//     --iree-codegen-llvmcpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cpu/matmul_step_03_tile_and_vectorize_parallel.mlir | \
//   time iree-run-module --function=matmul_static \
//   --device=local-task --task_topology_group_count=16 \
//     --input="1024x12345xf32=1" \
//     --input="12345x4096xf32=1" \
//     --input="1024x4096xf32=0" | \
//   FileCheck ${IREE_SAMPLES_DIR}/transform_dialect/examples/cpu/matmul_step_03_tile_and_vectorize_parallel.mlir
// ```
//
// The time command prints something resembling:
// `13.58user 0.07system 0:01.74elapsed 784%CPU (0avgtext+0avgdata 267008maxresident)k`
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
// CHECK-IR:     tensor.extract_slice {{.*}} [128, 128] [1, 1] : tensor<1024x4096xf32> to tensor<128x128xf32>
// CHECK-IR:     scf.for {{.*}} -> (tensor<128x128xf32>) {
// CHECK-IR:       scf.for {{.*}} -> (tensor<128x128xf32>) {
// CHECK-IR:         scf.for {{.*}} -> (tensor<128x128xf32>) {
// CHECK-IR:           vector.transfer_read {{.*}} {in_bounds = [true, true]} : tensor<128x128xf32>, vector<16x8xf32>
// CHECK-IR:           scf.for {{.*}} -> (vector<16x8xf32>) {
// CHECK-IR:             vector.transfer_read {{.*}} {in_bounds = [true, true]} : tensor<1024x2048xf32>, vector<16x1xf32>
// CHECK-IR:             vector.transfer_read {{.*}} {in_bounds = [true, true]} : tensor<2048x4096xf32>, vector<1x8xf32>
// CHECK-IR:             vector.contract {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"], kind = #vector.kind<add>} 
// CHECK-IR-SAME:          : vector<16x1xf32>, vector<1x8xf32> into vector<16x8xf32>
// CHECK-IR:             scf.yield {{.*}} : vector<16x8xf32>
// CHECK-IR:           }
// CHECK-IR:           vector.transfer_write {{.*}} {in_bounds = [true, true]} : vector<16x8xf32>, tensor<128x128xf32>
// CHECK-IR:           scf.yield {{.*}} : tensor<128x128xf32>
// CHECK-IR:         }
// CHECK-IR:         scf.yield {{.*}} : tensor<128x128xf32>
// CHECK-IR:       }
// CHECK-IR:       scf.yield {{.*}} : tensor<128x128xf32>
// CHECK-IR:     }
// CHECK-IR:     tensor.insert_slice {{.*}} into {{.*}} [128, 128] [1, 1] : tensor<128x128xf32> into tensor<1024x4096xf32>
// CHECK-IR:     scf.yield %inserted_slice : tensor<1024x4096xf32>
// CHECK-IR:   }
// CHECK-IR:   scf.yield {{.*}} : tensor<1024x4096xf32>

// Comment parts of the IR below starting from the bottom-up and rerun the command
// to see the effects of step 1. ; step 1. + step 2.; etc..
transform.structured.canonicalized_sequence failures(propagate) {
// transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %original_matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  // IREE-specific connection to the runtime and threadpool, required to run e2e.
  // ============================================================================
  %forall, %matmul =
    transform.iree.tile_to_forall_and_workgroup_count_region %original_matmul 
      tile_sizes [256, 256]
      // TODO: IREE needs own workgroup mapping attribute independent of GPU.
      ( mapping = [#gpu.block<x>, #gpu.block<y>] )

  // Step 1. Tile to forall and sequential scf.for.
  // ======================================================
  %matmul_l1, %loops_l1:3 = transform.structured.tile_to_scf_for %matmul [128, 128, 512]
  
  // Step 2. Tile to forall and sequential scf.for.
  // ======================================================
  %matmul_l2, %loops_l2:3 = transform.structured.tile_to_scf_for %matmul_l1 [8, 16, 1]

  // Step 3. Vectorize.
  // ======================================================
  %func = transform.structured.match ops{["func.func"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  transform.structured.vectorize %func

  // IREE-specific bufferization.
  // ============================================================================
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
}
