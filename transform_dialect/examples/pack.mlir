!TYPE_t = tensor<${N}x${M}xf32>
!PACKED_TYPE_t = tensor<1x1x${Npad}x${Mpad}xf32>

func.func @pack_static(%arg: !TYPE_t) -> !PACKED_TYPE_t {
  %empty = tensor.empty() : !PACKED_TYPE_t
  %cst = arith.constant 0.000000e+00 : f32

  %pack = tensor.pack %arg padding_value(%cst : f32)
    inner_dims_pos = [0, 1] inner_tiles = [${Npad}, ${Mpad}] into %empty
    : !TYPE_t -> !PACKED_TYPE_t

  return %pack : !PACKED_TYPE_t
}
