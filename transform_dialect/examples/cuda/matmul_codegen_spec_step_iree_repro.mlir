// Instructions; TL;DR
// ===================
//
// This script shows an example of script that aims at reproducing the IREE pass
// pipeline behavior.
//
// This is dependent on the integration of the prerequisites in:
//   https://github.com/nicolasvasilache/iree/commits/matmul-cuda-pad
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   cat ${IREE_SAMPLES_DIR}/transform_dialect/examples/matmul.mlir |\
//   sed "s/\${M}/1024/g" | sed "s/\${K}/2048/g" | sed "s/\${N}/4096/g" | \
//   ${IREE_DIR}/build/tools/iree-opt \
//     --iree-hal-target-backends=cuda \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_DIR}/build/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmgpu-lower-executable-target)))' \
//      --iree-codegen-llvmgpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/cuda/matmul_codegen_spec_step_iree_repro.mlir \
//      --iree-codegen-llvmgpu-enable-transform-dialect-jit=false | \
//   FileCheck transform_dialect/examples/cuda/matmul_codegen_spec_step_iree_repro.mlir
// ```

// Extracting the relevant parts of the IREE pass pipeline in comments.
//
// addGPUMatmulTensorCorePassPipeline(OpPassManager &pm, unsigned pipelineDepth)
//   tileAndBufferize(pm);
//   createLLVMGPUTileAndDistribute(/*distributeToWarp=*/true));
//   createGPUMultiBuffering(pipelineDepth));
//   createWorkGroupSwizzle(logSwizzleTile));
//   // Linalg -> vector
//   createLLVMGPUTensorCoreVectorizationPass());
//   createOptimizeVectorTransferPass());
//   // Distribute shared memory copies.
//   createGPUDistributeSharedMemoryCopy());
//   if (!llvmgpuUseMMASync) createGPUReduceSharedMemoryBankConflicts());
//   // Vector -> MMA ops
//   memref::createFoldMemRefAliasOpsPass());
//   createLLVMGPUVectorToGPU());
//   // Pipeline memory operations.
//   createGPUPipeliningPass(/*epiloguePeeling=*/false, pipelineDepth));

// addGPUMatmulSimtPassPipeline(OpPassManager &pm) {
//   tileAndDistributeToWorkgroup(pm);
//   createLLVMGPUTensorAlloc());     // this is transform.iree.promote_operands
//   createLLVMGPUTileTensor(false)); // tile to scf.for to reduce local sizes.
//   // Linalg -> vector
//   createGPUVectorizationPass());
//   // tensor to memref
//   addBufferizePasses(nestedModulePM);
//   // distribute foreach threads
//   createLLVMGPUDistribute());
//   createGPUDistributeSharedMemoryCopy());
//   createGPUReduceSharedMemoryBankConflicts());
//   createWorkGroupSwizzle(logSwizzleTile));
//   // Even though we vectorize before bufferization we are not able to hoist
//   // accumulator load/store out of the K loop until distribution. Therefore we
//   // still rely on buffer level transformations for transfer ops hoisting and
//   // store to load forwarding. This relies on shacky alias analysis and we need
//   // to move this to tensor level once we have better abstractions.
//   createOptimizeVectorTransferPass());
//   // Pipeline memory operations.
//   createGPUPipeliningPass());

// Implement the beginning of addGPUMatmulSimtPassPipeline
transform.structured.canonicalized_sequence failures(propagate) {
// transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  // Step 1. Tile to foralls.
  // ======================================================
  %forall_l1, %matmul_l1 =
    transform.iree.tile_to_forall_and_workgroup_count_region %matmul tile_sizes [32, 128]
      ( mapping = [#gpu.block<y>, #gpu.block<x>] )

  // Step 2. Tile reduction to scf.for to reduce local sizes.
  // ========================================================
  %matmul_l2, %loops:1 = transform.structured.tile_to_scf_for %matmul_l1 [0, 0, 32]

  // Step 3. Promote operands to shared memory.
  // ==========================================
  // This is createLLVMGPUTensorAlloc, it allocates larger tensors before a further
  // level of tiling.
  %promoted_matmul_l2, %alloc_1 , %alloc_2 = transform.iree.promote_operands %matmul_l2 [0, 1] 
    : (!pdl.operation) -> (!pdl.operation, !pdl.operation, !pdl.operation)

  // Step 4. Tile to foralls.
  // ======================================================
  %forall_l3, %matmul_l3 =
    transform.structured.tile_to_forall_op %promoted_matmul_l2 num_threads [8, 32]
      ( mapping = [#gpu.thread<y>, #gpu.thread<x>] )

  // TODO: further steps still need to be connected.
}
