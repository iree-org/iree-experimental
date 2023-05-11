!TYPE_t = tensor<1x1x${Npad}x${Mpad}xf32>
!UNPACKED_TYPE_t = tensor<${N}x${M}xf32>

func.func @unpack_static(%arg: !TYPE_t) -> !UNPACKED_TYPE_t {
  %empty = tensor.empty() : !UNPACKED_TYPE_t

  %unpack = tensor.unpack %arg
    inner_dims_pos = [0, 1] inner_tiles = [${Npad}, ${Mpad}] into %empty
    : !TYPE_t -> !UNPACKED_TYPE_t

  return %unpack : !UNPACKED_TYPE_t
}
