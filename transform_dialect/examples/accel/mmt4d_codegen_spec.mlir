// This script shows an example lowering matmul through IREE for a special accelerator.
//
// ```
//   export IREE_DIR=${HOME}/github/iree; \
//   export IREE_SAMPLES_DIR=${HOME}/github/iree-samples; \
//   ${IREE_DIR}/build/tools/iree-opt \
//     ${IREE_SAMPLES_DIR}/transform_dialect/examples/accel/mmt4d_source.mlir \
//     --iree-hal-target-backends=llvm-cpu \
//     --iree-abi-transformation-pipeline \
//     --iree-flow-transformation-pipeline \
//     --iree-stream-transformation-pipeline \
//     --iree-hal-configuration-pipeline | \
//   ${IREE_DIR}/build/tools/iree-opt \
//      --pass-pipeline='builtin.module(hal.executable(hal.executable.variant(iree-llvmcpu-lower-executable-target, builtin.module(func.func(iree-hoist-statically-bound-allocations)))))' \
//      --iree-codegen-llvmcpu-use-transform-dialect=${IREE_SAMPLES_DIR}/transform_dialect/examples/accel/mmt4d_codegen_spec.mlir
// ```
//
// We would expect IR resembling:
//
// CHECK-IR: #map = affine_map<(d0) -> (d0 * 16)>
// CHECK-IR: #map1 = affine_map<(d0) -> (d0 * 32)>
// CHECK-IR: #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-IR: #map3 = affine_map<(d0) -> (d0 * 8)>
// CHECK-IR: %alloca = memref.alloca() {alignment = 64 : i64} : memref<8x8x8x8xf32>
// CHECK-IR: %alloca_0 = memref.alloca() {alignment = 64 : i64} : memref<32x8x8x1xf32>
// CHECK-IR: %alloca_1 = memref.alloca() {alignment = 64 : i64} : memref<16x8x8x1xf32>
// CHECK-IR: affine.apply #map(%workgroup_id_y)
// CHECK-IR: affine.apply #map(%workgroup_id_x)
// CHECK-IR: memref.subview {{.*}} : memref<48x384x8x1xf32> to memref<16x384x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>
// CHECK-IR: memref.subview {{.*}} : memref<64x384x8x1xf32> to memref<32x384x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>
// CHECK-IR: memref.subview {{.*}} : memref<48x64x8x8xf32> to memref<16x32x8x8xf32, strided<[4096, 64, 8, 1], offset: ?>>
// CHECK-IR: scf.for {{.*}} {
// CHECK-IR:   memref.subview {{.*}} : memref<16x384x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>> to memref<16x8x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>
// CHECK-IR:   memref.subview {{.*}} : memref<32x384x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>> to memref<32x8x8x1xf32, strided<[3072, 8, 1, 1], offset: ?>>
// CHECK-IR:   linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins{{.*}} outs(%alloca_1 : memref<16x8x8x1xf32>) { linalg.yield }
// CHECK-IR:   linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins{{.*}} outs(%alloca_0 : memref<32x8x8x1xf32) { linalg.yield }
// CHECK-IR:   gpu.thread_id  x
// CHECK-IR:   gpu.thread_id  y
// CHECK-IR:   gpu.thread_id  z
// CHECK-IR:   affine.apply #map3(%6)
// CHECK-IR:   affine.apply #map3(%5)
// CHECK-IR:   memref.subview %alloca_1 : memref<16x8x8x1xf32> to memref<8x8x8x1xf32, strided<[64, 8, 1, 1], offset: ?>>
// CHECK-IR:   memref.subview %alloca_0 : memref<32x8x8x1xf32> to memref<8x8x8x1xf32, strided<[64, 8, 1, 1], offset: ?>>
// CHECK-IR:   memref.subview {{.*}} : memref<16x32x8x8xf32, strided<[4096, 64, 8, 1], offset: ?>> to memref<8x8x8x8xf32, strided<[4096, 64, 8, 1], offset: ?>>
// CHECK-IR:   linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins{{.*}} outs(%alloca : memref<8x8x8x8xf32>) { linalg.yield }
// CHECK-IR:   linalg.mmt4d ins({{.*}}, {{.*}} : memref<8x8x8x1xf32, strided<[64, 8, 1, 1], offset: ?>>, memref<8x8x8x1xf32, strided<[64, 8, 1, 1], offset: ?>>) outs(%alloca : memref<8x8x8x8xf32>)
// CHECK-IR:   linalg.generic {indexing_maps = [#map2, #map2], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%alloca : memref<8x8x8x8xf32>) outs({{.*}} : memref<8x8x8x8xf32, strided<[4096, 64, 8, 1], offset: ?>>) { linalg.yield }
// CHECK-IR:   gpu.barrier
// CHECK-IR:  }
// CHECK-IR:  return
// CHECK-IR: }

module attributes { transform.with_named_sequence } {
  transform.named_sequence @cleanup(%variant_op: !transform.any_op {transform.readonly}) {
    %func = transform.structured.match ops{["func.func"]} in %variant_op : (!transform.any_op) -> !transform.any_op
      transform.apply_patterns to %func {
      transform.apply_patterns.linalg.tiling_canonicalization
      transform.apply_patterns.iree.fold_fill_into_pad
      transform.apply_patterns.scf.for_loop_canonicalization
      transform.apply_patterns.canonicalization
    } : !transform.any_op
    transform.iree.apply_licm %func : !transform.any_op
    transform.iree.apply_cse %func : !transform.any_op
    transform.yield
  }

  transform.sequence failures(propagate) {
  ^bb1(%variant_op: !transform.any_op):
    %matmul = transform.structured.match ops{["linalg.mmt4d"]} in %variant_op : (!transform.any_op) -> !transform.any_op

    // First level tile to forall with tile_sizes [16, 32].
    %forall, %tiled_matmul =
      transform.structured.tile_to_forall_op %matmul tile_sizes [16, 32]
        ( mapping = [#gpu.block<y>, #gpu.block<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
    transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall
      : (!transform.any_op) -> ()

    // Tile reduction dimension.
    %tiled_reduction, %loop =
      transform.structured.tile %tiled_matmul [0, 0, 8]
      : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pad operation.
    %padded, %pad, %__ = transform.structured.pad %tiled_reduction {
      padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[1, 1, 0],
      copy_back_op="none"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Second level tile to forall with tile_sizes [8, 8].
    %forall_1, %tiled_matmul_1 =
      transform.structured.tile_to_forall_op %padded tile_sizes [8, 8]
        ( mapping = [#gpu.thread<y>, #gpu.thread<x>] ) : (!transform.any_op) -> (!transform.any_op, !transform.any_op)

    // Pad operation.
    %padded_1, %pad_1, %_ = transform.structured.pad %tiled_matmul_1 {
      padding_values=[0.0 : f32, 0.0 : f32, 0.0 : f32],
      padding_dimensions=[0, 1, 2],
      pack_paddings=[0, 0, 1],
      copy_back_op="none"
    } : (!transform.any_op) -> (!transform.any_op, !transform.any_op, !transform.any_op)

    // Clean up.
    transform.include @cleanup failures(propagate) (%variant_op) : (!transform.any_op) -> ()
    transform.iree.eliminate_empty_tensors %variant_op : (!transform.any_op) -> ()

    // Bufferize and drop HAL decriptor from memref ops.
    %variant_op_3 = transform.iree.bufferize %variant_op : (!transform.any_op) -> !transform.any_op

    // Post-bufferization mapping workgroup.
    %memref_func = transform.structured.match ops{["func.func"]} in %variant_op_3 : (!transform.any_op) -> !transform.any_op
    transform.iree.forall_to_workgroup %memref_func : (!transform.any_op) -> ()
    transform.iree.map_nested_forall_to_gpu_threads %memref_func workgroup_dims = [4, 2, 1] subgroup_size = 1 : (!transform.any_op) -> ()
    transform.iree.hoist_static_alloc %memref_func : (!transform.any_op) -> ()
  }
}
