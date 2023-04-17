!I_1d_t = tensor<${SZ1}xf32>
!O_1d_t = tensor<${SZ1}xf32>

func.func private @pointwise_1d_static(
    %I : !I_1d_t, %O : !O_1d_t) -> !O_1d_t {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>,
                                        affine_map<(d0) -> (d0)>],
  iterator_types = ["parallel"]}
    ins(%I : !I_1d_t) outs(%O: !O_1d_t) {
      ^bb0(%in0: f32, %out0: f32):
        %re = arith.mulf %out0, %in0 : f32
        linalg.yield %re : f32
    } -> !O_1d_t

  return %0 : !O_1d_t
}

!I_2d_t = tensor<${SZ1}x${SZ2}xf32>
!O_2d_t = tensor<${SZ1}x${SZ2}xf32>

func.func private @pointwise_2d_static(
    %I : !I_2d_t, %O : !O_2d_t) -> !O_2d_t {
  %0 = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                        affine_map<(d0, d1) -> (d0, d1)>],
  iterator_types = ["parallel", "parallel"]}
    ins(%I : !I_2d_t) outs(%O: !O_2d_t) {
      ^bb0(%in0: f32, %out0: f32):
        %re = arith.mulf %out0, %in0 : f32
        linalg.yield %re : f32
    } -> !O_2d_t

  return %0 : !O_2d_t
}
