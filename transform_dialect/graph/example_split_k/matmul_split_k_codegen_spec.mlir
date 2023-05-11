//
// Instructions
// ============
//
// See transform_dialect/graph/example_split_k/matmul_split_k_graph_spec.mlir
//

transform.sequence failures(propagate) {
^bb1(%variant_op: !pdl.operation):
  %toplevel_forall = transform.structured.match ops{["scf.forall"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  transform.iree.populate_workgroup_count_region_using_num_threads_slice %toplevel_forall
    : (!pdl.operation) -> ()

  // TODO: apply the rest of codegen. In practice, this would just use whatever
  // IREE already provides, once nested block-level scf.forall work properly.
  // Here we apply just the first step tiling to block.x/y to demonstrate what
  // is missing at the boundary when mapping nested scf.forall to blocks.

  // Tile to forall and sequential scf.for.
  // ======================================
  %meatier_matmul = transform.structured.match ops{["linalg.matmul"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation
  %forall_l1, %matmul_l1 = transform.structured.tile_to_forall_op %meatier_matmul tile_sizes [128, 128]
      ( mapping = [#gpu.block<y>, #gpu.block<x>] )
  %matmul_l2, %loop = transform.structured.tile_to_scf_for %matmul_l1 [0, 0, 16]
  // Post-tiling canonicalizations and cleanups.
  transform.iree.apply_patterns %variant_op 
    {canonicalization, cse, licm, tiling_canonicalization}
      : (!pdl.operation) -> ()

  // FIXME: This does not seem robust enough to be applied twice atm.
  // We expect [4, 1, 77] but get [1, 1, 77] instead.
  // transform.iree.populate_workgroup_count_region_using_num_threads_slice %forall_l1
  //   : (!pdl.operation) -> ()

  %func = transform.structured.match ops{["func.func"]} in %variant_op
    : (!pdl.operation) -> !pdl.operation

  // FIXME: This currently does not work with nested block-mapped loops.
  // Fixing this is an opportunity to unify block and thread mapping too.
  // transform.iree.forall_to_workgroup %func : (!pdl.operation) -> ()

  transform.iree.map_nested_forall_to_gpu_threads %func
      workgroup_dims = [1, 1, 1] : (!pdl.operation) -> ()
}
