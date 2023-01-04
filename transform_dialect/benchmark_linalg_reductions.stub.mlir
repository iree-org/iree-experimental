////////////////////////////////////////////////////////////////////////////////
// Static stuff
////////////////////////////////////////////////////////////////////////////////

///////////////// 2-D
#map_ij_to_ij = affine_map<(d0, d1) -> (d0, d1)>
#map_ij_to_i = affine_map<(d0, d1) -> (d0)>
#iters_par_red = ["parallel", "reduction"]
#iters_par_par = ["parallel", "parallel"]
#trait_reduction_2d = {
    indexing_maps = [#map_ij_to_ij, #map_ij_to_i],
    iterator_types = #iters_par_red
}
#trait_broadcast_elementwise_2d = {
    indexing_maps = [#map_ij_to_ij, #map_ij_to_i, #map_ij_to_ij],
    iterator_types = #iters_par_par
}

!in_tensor_reduction_2d_static_t = tensor<${SZ1}x${SZ2}x${ELEMENTAL_TYPE}>
!out_tensor_reduction_2d_static_t = tensor<${SZ1}x${ELEMENTAL_TYPE}>

func.func private @reduction_2d_static(%input : !in_tensor_reduction_2d_static_t)
     -> (!out_tensor_reduction_2d_static_t) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant ${ZERO} : ${ELEMENTAL_TYPE}

  %tmp_empty = tensor.empty() : !out_tensor_reduction_2d_static_t
  %tmp_filled = linalg.fill ins(%cst : ${ELEMENTAL_TYPE}) outs(%tmp_empty : !out_tensor_reduction_2d_static_t)
     ->   !out_tensor_reduction_2d_static_t
  %tmp_reduced = linalg.generic #trait_reduction_2d
    ins(%input : !in_tensor_reduction_2d_static_t)
   outs(%tmp_filled : !out_tensor_reduction_2d_static_t) {
      ^bb0(%a: ${ELEMENTAL_TYPE}, %b: ${ELEMENTAL_TYPE}):
        %3 = ${ADD_OP} %a, %b : ${ELEMENTAL_TYPE}
        linalg.yield %3 : ${ELEMENTAL_TYPE}
      } -> !out_tensor_reduction_2d_static_t
  return %tmp_reduced : !out_tensor_reduction_2d_static_t
}

func.func private @reduction_2d_elementwise_static(%input : !in_tensor_reduction_2d_static_t) 
    -> (!in_tensor_reduction_2d_static_t) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant ${ZERO} : ${ELEMENTAL_TYPE}

  %tmp_empty = tensor.empty() : !out_tensor_reduction_2d_static_t
  %tmp_filled = linalg.fill ins(%cst : ${ELEMENTAL_TYPE}) outs(%tmp_empty : !out_tensor_reduction_2d_static_t) 
    ->   !out_tensor_reduction_2d_static_t
  %tmp_reduced = linalg.generic #trait_reduction_2d
    ins(%input : !in_tensor_reduction_2d_static_t)
   outs(%tmp_filled : !out_tensor_reduction_2d_static_t) {
      ^bb0(%a: ${ELEMENTAL_TYPE}, %b: ${ELEMENTAL_TYPE}):
        %3 = ${ADD_OP} %a, %b : ${ELEMENTAL_TYPE}
        linalg.yield %3 : ${ELEMENTAL_TYPE}
      } -> !out_tensor_reduction_2d_static_t

  %result_empty = tensor.empty() : !in_tensor_reduction_2d_static_t
  %3 = linalg.generic #trait_broadcast_elementwise_2d
    ins(%input, %tmp_reduced : !in_tensor_reduction_2d_static_t, !out_tensor_reduction_2d_static_t)
   outs(%result_empty : !in_tensor_reduction_2d_static_t) {
      ^bb0(%a: ${ELEMENTAL_TYPE}, %b: ${ELEMENTAL_TYPE}, %c: ${ELEMENTAL_TYPE}):
        %tmp_filled2 = ${DIV_OP} %a, %b : ${ELEMENTAL_TYPE}
        linalg.yield %tmp_filled2 : ${ELEMENTAL_TYPE}
      } -> !in_tensor_reduction_2d_static_t

  return %3 : !in_tensor_reduction_2d_static_t
}

///////////////// 3-D
#map_ijk_to_ijk = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map_ijk_to_ij = affine_map<(d0, d1, d2) -> (d0, d1)>
#iters_2_par_1_red = ["parallel", "parallel", "reduction"]
#iters_3_par = ["parallel", "parallel", "parallel"]
#trait_reduction_3d = {
    indexing_maps = [#map_ijk_to_ijk, #map_ijk_to_ij],
    iterator_types = #iters_2_par_1_red
}
#trait_broadcast_elementwise_3d = {
    indexing_maps = [#map_ijk_to_ijk, #map_ijk_to_ij, #map_ijk_to_ijk],
    iterator_types = #iters_3_par
}

!in_tensor_reduction_3d_static_t = tensor<${SZ1}x${SZ2}x${SZ3}x${ELEMENTAL_TYPE}>
!out_tensor_reduction_3d_static_t = tensor<${SZ1}x${SZ2}x${ELEMENTAL_TYPE}>

func.func private @reduction_3d_static(%input : !in_tensor_reduction_3d_static_t) 
    -> (!out_tensor_reduction_3d_static_t) {
  %c0 = arith.constant 0 : index
  %cst = arith.constant ${ZERO} : ${ELEMENTAL_TYPE}

  %tmp_empty = tensor.empty() : !out_tensor_reduction_3d_static_t
  %tmp_filled = linalg.fill ins(%cst : ${ELEMENTAL_TYPE}) outs(%tmp_empty : !out_tensor_reduction_3d_static_t) 
    ->   !out_tensor_reduction_3d_static_t
  %tmp_reduced = linalg.generic #trait_reduction_3d
    ins(%input : !in_tensor_reduction_3d_static_t)
   outs(%tmp_filled : !out_tensor_reduction_3d_static_t) {
      ^bb0(%a: ${ELEMENTAL_TYPE}, %b: ${ELEMENTAL_TYPE}):
        %3 = ${ADD_OP} %a, %b : ${ELEMENTAL_TYPE}
        linalg.yield %3 : ${ELEMENTAL_TYPE}
      } -> !out_tensor_reduction_3d_static_t
  return %tmp_reduced : !out_tensor_reduction_3d_static_t
}

func.func private @reduction_3d_elementwise_static(%input : !in_tensor_reduction_3d_static_t) 
    -> (!in_tensor_reduction_3d_static_t) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant ${ZERO} : ${ELEMENTAL_TYPE}

  %tmp_empty = tensor.empty() : !out_tensor_reduction_3d_static_t
  %tmp_filled = linalg.fill ins(%cst : ${ELEMENTAL_TYPE}) outs(%tmp_empty : !out_tensor_reduction_3d_static_t) 
    ->   !out_tensor_reduction_3d_static_t
  %tmp_reduced = linalg.generic #trait_reduction_3d
    ins(%input : !in_tensor_reduction_3d_static_t)
   outs(%tmp_filled : !out_tensor_reduction_3d_static_t) {
      ^bb0(%a: ${ELEMENTAL_TYPE}, %b: ${ELEMENTAL_TYPE}):
        %3 = ${ADD_OP} %a, %b : ${ELEMENTAL_TYPE}
        linalg.yield %3 : ${ELEMENTAL_TYPE}
      } -> !out_tensor_reduction_3d_static_t

  %result_empty = tensor.empty() : !in_tensor_reduction_3d_static_t
  %3 = linalg.generic #trait_broadcast_elementwise_3d
    ins(%input, %tmp_reduced: !in_tensor_reduction_3d_static_t, !out_tensor_reduction_3d_static_t)
   outs(%result_empty : !in_tensor_reduction_3d_static_t) {
      ^bb0(%a: ${ELEMENTAL_TYPE}, %b: ${ELEMENTAL_TYPE}, %c: ${ELEMENTAL_TYPE}):
        %tmp_filled2 = ${DIV_OP} %a, %b : ${ELEMENTAL_TYPE}
        linalg.yield %tmp_filled2 : ${ELEMENTAL_TYPE}
      } -> !in_tensor_reduction_3d_static_t

  return %3 : !in_tensor_reduction_3d_static_t
}

////////////////////////////////////////////////////////////////////////////////
// Dynamic stuff
////////////////////////////////////////////////////////////////////////////////

!in_tensor_reduction_2d_dynamic_t = tensor<?x?x${ELEMENTAL_TYPE}>
!out_tensor_reduction_2d_dynamic_t = tensor<?x${ELEMENTAL_TYPE}>

func.func private @reduction_2d_dynamic(%input : !in_tensor_reduction_2d_dynamic_t) 
    -> (!out_tensor_reduction_2d_dynamic_t) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 0 : index
  %cst = arith.constant ${ZERO} : ${ELEMENTAL_TYPE}

  %d0 = tensor.dim %input, %c0 : !in_tensor_reduction_2d_dynamic_t
  %d1 = tensor.dim %input, %c1 : !in_tensor_reduction_2d_dynamic_t
  %tmp_empty = tensor.empty(%d0) : !out_tensor_reduction_2d_dynamic_t
  %tmp_filled = linalg.fill ins(%cst : ${ELEMENTAL_TYPE}) outs(%tmp_empty : !out_tensor_reduction_2d_dynamic_t) 
    ->   !out_tensor_reduction_2d_dynamic_t
  %tmp_reduced = linalg.generic #trait_reduction_2d
    ins(%input : !in_tensor_reduction_2d_dynamic_t)
   outs(%tmp_filled : !out_tensor_reduction_2d_dynamic_t) {
      ^bb0(%a: ${ELEMENTAL_TYPE}, %b: ${ELEMENTAL_TYPE}):
        %3 = ${ADD_OP} %a, %b : ${ELEMENTAL_TYPE}
        linalg.yield %3 : ${ELEMENTAL_TYPE}
      } -> !out_tensor_reduction_2d_dynamic_t
  return %tmp_reduced : !out_tensor_reduction_2d_dynamic_t
}

func.func private @reduction_2d_elementwise_dynamic(%input : !in_tensor_reduction_2d_dynamic_t) 
    -> (!in_tensor_reduction_2d_dynamic_t) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %cst = arith.constant ${ZERO} : ${ELEMENTAL_TYPE}

  %d0 = tensor.dim %input, %c0 : !in_tensor_reduction_2d_dynamic_t
  %d1 = tensor.dim %input, %c1 : !in_tensor_reduction_2d_dynamic_t
  %tmp_empty = tensor.empty(%d0) : !out_tensor_reduction_2d_dynamic_t
  %tmp_filled = linalg.fill ins(%cst : ${ELEMENTAL_TYPE}) outs(%tmp_empty : !out_tensor_reduction_2d_dynamic_t) 
    ->   !out_tensor_reduction_2d_dynamic_t
  %tmp_reduced = linalg.generic #trait_reduction_2d
    ins(%input : !in_tensor_reduction_2d_dynamic_t) 
   outs(%tmp_filled : !out_tensor_reduction_2d_dynamic_t) {
      ^bb0(%a: ${ELEMENTAL_TYPE}, %b: ${ELEMENTAL_TYPE}):
        %3 = ${ADD_OP} %a, %b : ${ELEMENTAL_TYPE}
        linalg.yield %3 : ${ELEMENTAL_TYPE}
      } -> !out_tensor_reduction_2d_dynamic_t

  %result_empty = tensor.empty(%d0, %d1) : !in_tensor_reduction_2d_dynamic_t
  %3 = linalg.generic #trait_broadcast_elementwise_2d
    ins(%input, %tmp_reduced : !in_tensor_reduction_2d_dynamic_t, !out_tensor_reduction_2d_dynamic_t) 
   outs(%result_empty : !in_tensor_reduction_2d_dynamic_t) {
      ^bb0(%a: ${ELEMENTAL_TYPE}, %b: ${ELEMENTAL_TYPE}, %c: ${ELEMENTAL_TYPE}):
        %tmp_filled2 = ${DIV_OP} %a, %b : ${ELEMENTAL_TYPE}
        linalg.yield %tmp_filled2 : ${ELEMENTAL_TYPE}
      } -> !in_tensor_reduction_2d_dynamic_t

  return %3 : !in_tensor_reduction_2d_dynamic_t
}
