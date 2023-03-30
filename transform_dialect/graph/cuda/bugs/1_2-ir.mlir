// Repro: iree-compile --iree-hal-target-backends=cuda <this-file>
// Repro: iree-compile --iree-hal-target-backends=llvm-cpu <this-file>
 
 #map = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d2, d3, d5)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d2, d1, d4, d5)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d3, d4)>
module {
  func.func @fill_matmul_static(%arg0: tensor<999x1999xf32>, %arg1: tensor<1999x3999xf32>, %arg2: tensor<999x3999xf32>) -> tensor<999x3999xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<999x3999xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<999x3999xf32>) -> tensor<999x3999xf32>
    %2 = tensor.empty() : tensor<1x1x1024x2016xf32>
    %cst_0 = arith.constant 0.000000e+00 : f32
    %pack = tensor.pack %arg0 padding_value(%cst_0 : f32) inner_dims_pos = [0, 1] inner_tiles = [1024, 2016] into %2 : tensor<999x1999xf32> -> tensor<1x1x1024x2016xf32>
    %3 = tensor.empty() : tensor<1x1x4096x2016xf32>
    %cst_1 = arith.constant 0.000000e+00 : f32
    %pack_2 = tensor.pack %arg1 padding_value(%cst_1 : f32) inner_dims_pos = [1, 0] inner_tiles = [4096, 2016] into %3 : tensor<1999x3999xf32> -> tensor<1x1x4096x2016xf32>
    %4 = tensor.empty() : tensor<1x1x1024x4096xf32>
    %cst_3 = arith.constant 0.000000e+00 : f32
    %pack_4 = tensor.pack %1 padding_value(%cst_3 : f32) inner_dims_pos = [0, 1] inner_tiles = [1024, 4096] into %4 : tensor<999x3999xf32> -> tensor<1x1x1024x4096xf32>
    %5 = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_2 : tensor<1x1x1024x2016xf32>, tensor<1x1x4096x2016xf32>) outs(%pack_4 : tensor<1x1x1024x4096xf32>) {
    ^bb0(%in: f32, %in_5: f32, %out: f32):
      %6 = arith.mulf %in, %in_5 : f32
      %7 = arith.addf %out, %6 : f32
      linalg.yield %7 : f32
    } -> tensor<1x1x1024x4096xf32>
    %unpack = tensor.unpack %5 inner_dims_pos = [0, 1] inner_tiles = [1024, 4096] into %1 : tensor<1x1x1024x4096xf32> -> tensor<999x3999xf32>
    return %unpack : tensor<999x3999xf32>
  }
}

