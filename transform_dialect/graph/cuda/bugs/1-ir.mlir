// Repro: iree-compile --iree-hal-target-backends=cuda <this-file>
 
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4, d6)>
#map5 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d3, d2, d5, d6)>
#map6 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4, d5)>
#map7 = affine_map<(d0, d1, d2) -> (d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map11 = affine_map<(d0, d1, d2, d3) -> ()>
#map12 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
module {
  func.func @forward(%arg0: tensor<3x17x768xf32>) -> tensor<3x17x768xf32> {
    %cst = arith.constant dense<8.000000e+00> : tensor<f64>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<768x768xf32>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<768xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %cst_3 = arith.constant -3.40282347E+38 : f32
    %0 = tensor.empty() : tensor<768x768xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<768x768xf32>) outs(%0 : tensor<768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<768x768xf32>
    %2 = tensor.empty() : tensor<3x768x768xf32>
    %3 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<768x768xf32>) outs(%2 : tensor<3x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x768x768xf32>
    %4 = tensor.empty() : tensor<3x17x768xf32>
    %5 = linalg.fill ins(%cst_2 : f32) outs(%4 : tensor<3x17x768xf32>) -> tensor<3x17x768xf32>
    %6 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %cst_4 = arith.constant 0.000000e+00 : f32
    %pack = tensor.pack %arg0 padding_value(%cst_4 : f32) inner_dims_pos = [1, 2] inner_tiles = [32, 768] into %6 : tensor<3x17x768xf32> -> tensor<3x1x1x32x768xf32>
    %7 = tensor.empty() : tensor<3x1x1x768x768xf32>
    %cst_5 = arith.constant 0.000000e+00 : f32
    %pack_6 = tensor.pack %3 padding_value(%cst_5 : f32) inner_dims_pos = [2, 1] inner_tiles = [768, 768] into %7 : tensor<3x768x768xf32> -> tensor<3x1x1x768x768xf32>
    %8 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %cst_7 = arith.constant 0.000000e+00 : f32
    %pack_8 = tensor.pack %5 padding_value(%cst_7 : f32) inner_dims_pos = [1, 2] inner_tiles = [32, 768] into %8 : tensor<3x17x768xf32> -> tensor<3x1x1x32x768xf32>
    %9 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack, %pack_6 : tensor<3x1x1x32x768xf32>, tensor<3x1x1x768x768xf32>) outs(%pack_8 : tensor<3x1x1x32x768xf32>) {
    ^bb0(%in: f32, %in_45: f32, %out: f32):
      %55 = arith.mulf %in, %in_45 : f32
      %56 = arith.addf %out, %55 : f32
      linalg.yield %56 : f32
    } -> tensor<3x1x1x32x768xf32>
    %unpack = tensor.unpack %9 inner_dims_pos = [1, 2] inner_tiles = [32, 768] into %5 : tensor<3x1x1x32x768xf32> -> tensor<3x17x768xf32>
    %10 = linalg.generic {indexing_maps = [#map3, #map7, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%unpack, %cst_1 : tensor<3x17x768xf32>, tensor<768xf32>) outs(%4 : tensor<3x17x768xf32>) {
    ^bb0(%in: f32, %in_45: f32, %out: f32):
      %55 = arith.addf %in, %in_45 : f32
      linalg.yield %55 : f32
    } -> tensor<3x17x768xf32>
    %11 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<768x768xf32>) outs(%0 : tensor<768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<768x768xf32>
    %12 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%11 : tensor<768x768xf32>) outs(%2 : tensor<3x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x768x768xf32>
    %13 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %cst_9 = arith.constant 0.000000e+00 : f32
    %pack_10 = tensor.pack %arg0 padding_value(%cst_9 : f32) inner_dims_pos = [1, 2] inner_tiles = [32, 768] into %13 : tensor<3x17x768xf32> -> tensor<3x1x1x32x768xf32>
    %14 = tensor.empty() : tensor<3x1x1x768x768xf32>
    %cst_11 = arith.constant 0.000000e+00 : f32
    %pack_12 = tensor.pack %12 padding_value(%cst_11 : f32) inner_dims_pos = [2, 1] inner_tiles = [768, 768] into %14 : tensor<3x768x768xf32> -> tensor<3x1x1x768x768xf32>
    %15 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %cst_13 = arith.constant 0.000000e+00 : f32
    %pack_14 = tensor.pack %5 padding_value(%cst_13 : f32) inner_dims_pos = [1, 2] inner_tiles = [32, 768] into %15 : tensor<3x17x768xf32> -> tensor<3x1x1x32x768xf32>
    %16 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_10, %pack_12 : tensor<3x1x1x32x768xf32>, tensor<3x1x1x768x768xf32>) outs(%pack_14 : tensor<3x1x1x32x768xf32>) {
    ^bb0(%in: f32, %in_45: f32, %out: f32):
      %55 = arith.mulf %in, %in_45 : f32
      %56 = arith.addf %out, %55 : f32
      linalg.yield %56 : f32
    } -> tensor<3x1x1x32x768xf32>
    %unpack_15 = tensor.unpack %16 inner_dims_pos = [1, 2] inner_tiles = [32, 768] into %5 : tensor<3x1x1x32x768xf32> -> tensor<3x17x768xf32>
    %17 = linalg.generic {indexing_maps = [#map3, #map7, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%unpack_15, %cst_1 : tensor<3x17x768xf32>, tensor<768xf32>) outs(%4 : tensor<3x17x768xf32>) {
    ^bb0(%in: f32, %in_45: f32, %out: f32):
      %55 = arith.addf %in, %in_45 : f32
      linalg.yield %55 : f32
    } -> tensor<3x17x768xf32>
    %expanded = tensor.expand_shape %17 [[0], [1], [2, 3]] : tensor<3x17x768xf32> into tensor<3x17x12x64xf32>
    %18 = tensor.empty() : tensor<3x12x17x64xf32>
    %19 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<3x17x12x64xf32>) outs(%18 : tensor<3x12x17x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x12x17x64xf32>
    %20 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<768x768xf32>) outs(%0 : tensor<768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<768x768xf32>
    %21 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%20 : tensor<768x768xf32>) outs(%2 : tensor<3x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x768x768xf32>
    %22 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %cst_16 = arith.constant 0.000000e+00 : f32
    %pack_17 = tensor.pack %arg0 padding_value(%cst_16 : f32) inner_dims_pos = [1, 2] inner_tiles = [32, 768] into %22 : tensor<3x17x768xf32> -> tensor<3x1x1x32x768xf32>
    %23 = tensor.empty() : tensor<3x1x1x768x768xf32>
    %cst_18 = arith.constant 0.000000e+00 : f32
    %pack_19 = tensor.pack %21 padding_value(%cst_18 : f32) inner_dims_pos = [2, 1] inner_tiles = [768, 768] into %23 : tensor<3x768x768xf32> -> tensor<3x1x1x768x768xf32>
    %24 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %cst_20 = arith.constant 0.000000e+00 : f32
    %pack_21 = tensor.pack %5 padding_value(%cst_20 : f32) inner_dims_pos = [1, 2] inner_tiles = [32, 768] into %24 : tensor<3x17x768xf32> -> tensor<3x1x1x32x768xf32>
    %25 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_17, %pack_19 : tensor<3x1x1x32x768xf32>, tensor<3x1x1x768x768xf32>) outs(%pack_21 : tensor<3x1x1x32x768xf32>) {
    ^bb0(%in: f32, %in_45: f32, %out: f32):
      %55 = arith.mulf %in, %in_45 : f32
      %56 = arith.addf %out, %55 : f32
      linalg.yield %56 : f32
    } -> tensor<3x1x1x32x768xf32>
    %unpack_22 = tensor.unpack %25 inner_dims_pos = [1, 2] inner_tiles = [32, 768] into %5 : tensor<3x1x1x32x768xf32> -> tensor<3x17x768xf32>
    %26 = linalg.generic {indexing_maps = [#map3, #map7, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%unpack_22, %cst_1 : tensor<3x17x768xf32>, tensor<768xf32>) outs(%4 : tensor<3x17x768xf32>) {
    ^bb0(%in: f32, %in_45: f32, %out: f32):
      %55 = arith.addf %in, %in_45 : f32
      linalg.yield %55 : f32
    } -> tensor<3x17x768xf32>
    %expanded_23 = tensor.expand_shape %26 [[0], [1], [2, 3]] : tensor<3x17x768xf32> into tensor<3x17x12x64xf32>
    %27 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_23 : tensor<3x17x12x64xf32>) outs(%18 : tensor<3x12x17x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x12x17x64xf32>
    %expanded_24 = tensor.expand_shape %10 [[0], [1], [2, 3]] : tensor<3x17x768xf32> into tensor<3x17x12x64xf32>
    %28 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_24 : tensor<3x17x12x64xf32>) outs(%18 : tensor<3x12x17x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x12x17x64xf32>
    %29 = tensor.empty() : tensor<3x12x64x17xf32>
    %30 = linalg.generic {indexing_maps = [#map8, #map10], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%19 : tensor<3x12x17x64xf32>) outs(%29 : tensor<3x12x64x17xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x12x64x17xf32>
    %collapsed = tensor.collapse_shape %28 [[0, 1], [2], [3]] : tensor<3x12x17x64xf32> into tensor<36x17x64xf32>
    %collapsed_25 = tensor.collapse_shape %30 [[0, 1], [2], [3]] : tensor<3x12x64x17xf32> into tensor<36x64x17xf32>
    %31 = tensor.empty() : tensor<36x17x17xf32>
    %32 = linalg.fill ins(%cst_2 : f32) outs(%31 : tensor<36x17x17xf32>) -> tensor<36x17x17xf32>
    %33 = tensor.empty() : tensor<36x1x1x32x64xf32>
    %cst_26 = arith.constant 0.000000e+00 : f32
    %pack_27 = tensor.pack %collapsed padding_value(%cst_26 : f32) inner_dims_pos = [1, 2] inner_tiles = [32, 64] into %33 : tensor<36x17x64xf32> -> tensor<36x1x1x32x64xf32>
    %34 = tensor.empty() : tensor<36x1x1x32x64xf32>
    %cst_28 = arith.constant 0.000000e+00 : f32
    %pack_29 = tensor.pack %collapsed_25 padding_value(%cst_28 : f32) inner_dims_pos = [2, 1] inner_tiles = [32, 64] into %34 : tensor<36x64x17xf32> -> tensor<36x1x1x32x64xf32>
    %35 = tensor.empty() : tensor<36x1x1x32x32xf32>
    %cst_30 = arith.constant 0.000000e+00 : f32
    %pack_31 = tensor.pack %32 padding_value(%cst_30 : f32) inner_dims_pos = [1, 2] inner_tiles = [32, 32] into %35 : tensor<36x17x17xf32> -> tensor<36x1x1x32x32xf32>
    %36 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_27, %pack_29 : tensor<36x1x1x32x64xf32>, tensor<36x1x1x32x64xf32>) outs(%pack_31 : tensor<36x1x1x32x32xf32>) {
    ^bb0(%in: f32, %in_45: f32, %out: f32):
      %55 = arith.mulf %in, %in_45 : f32
      %56 = arith.addf %out, %55 : f32
      linalg.yield %56 : f32
    } -> tensor<36x1x1x32x32xf32>
    %unpack_32 = tensor.unpack %36 inner_dims_pos = [1, 2] inner_tiles = [32, 32] into %32 : tensor<36x1x1x32x32xf32> -> tensor<36x17x17xf32>
    %expanded_33 = tensor.expand_shape %unpack_32 [[0, 1], [2], [3]] : tensor<36x17x17xf32> into tensor<3x12x17x17xf32>
    %37 = tensor.empty() : tensor<3x12x17x17xf32>
    %38 = linalg.generic {indexing_maps = [#map8, #map11, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_33, %cst : tensor<3x12x17x17xf32>, tensor<f64>) outs(%37 : tensor<3x12x17x17xf32>) {
    ^bb0(%in: f32, %in_45: f64, %out: f32):
      %55 = arith.truncf %in_45 : f64 to f32
      %56 = arith.divf %in, %55 : f32
      linalg.yield %56 : f32
    } -> tensor<3x12x17x17xf32>
    %39 = tensor.empty() : tensor<3x12x17x1xf32>
    %40 = linalg.fill ins(%cst_3 : f32) outs(%39 : tensor<3x12x17x1xf32>) -> tensor<3x12x17x1xf32>
    %41 = linalg.generic {indexing_maps = [#map8, #map12], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%38 : tensor<3x12x17x17xf32>) outs(%40 : tensor<3x12x17x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %55 = arith.maxf %in, %out : f32
      linalg.yield %55 : f32
    } -> tensor<3x12x17x1xf32>
    %42 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%38, %41 : tensor<3x12x17x17xf32>, tensor<3x12x17x1xf32>) outs(%37 : tensor<3x12x17x17xf32>) {
    ^bb0(%in: f32, %in_45: f32, %out: f32):
      %55 = arith.subf %in, %in_45 : f32
      linalg.yield %55 : f32
    } -> tensor<3x12x17x17xf32>
    %43 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%42 : tensor<3x12x17x17xf32>) outs(%37 : tensor<3x12x17x17xf32>) {
    ^bb0(%in: f32, %out: f32):
      %55 = math.exp %in : f32
      linalg.yield %55 : f32
    } -> tensor<3x12x17x17xf32>
    %44 = linalg.fill ins(%cst_2 : f32) outs(%39 : tensor<3x12x17x1xf32>) -> tensor<3x12x17x1xf32>
    %45 = linalg.generic {indexing_maps = [#map8, #map12], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%43 : tensor<3x12x17x17xf32>) outs(%44 : tensor<3x12x17x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %55 = arith.addf %in, %out : f32
      linalg.yield %55 : f32
    } -> tensor<3x12x17x1xf32>
    %46 = linalg.generic {indexing_maps = [#map8, #map12, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%43, %45 : tensor<3x12x17x17xf32>, tensor<3x12x17x1xf32>) outs(%37 : tensor<3x12x17x17xf32>) {
    ^bb0(%in: f32, %in_45: f32, %out: f32):
      %55 = arith.divf %in, %in_45 : f32
      linalg.yield %55 : f32
    } -> tensor<3x12x17x17xf32>
    %collapsed_34 = tensor.collapse_shape %46 [[0, 1], [2], [3]] : tensor<3x12x17x17xf32> into tensor<36x17x17xf32>
    %collapsed_35 = tensor.collapse_shape %27 [[0, 1], [2], [3]] : tensor<3x12x17x64xf32> into tensor<36x17x64xf32>
    %47 = tensor.empty() : tensor<36x17x64xf32>
    %48 = linalg.fill ins(%cst_2 : f32) outs(%47 : tensor<36x17x64xf32>) -> tensor<36x17x64xf32>
    %49 = tensor.empty() : tensor<36x1x1x32x32xf32>
    %cst_36 = arith.constant 0.000000e+00 : f32
    %pack_37 = tensor.pack %collapsed_34 padding_value(%cst_36 : f32) inner_dims_pos = [1, 2] inner_tiles = [32, 32] into %49 : tensor<36x17x17xf32> -> tensor<36x1x1x32x32xf32>
    %50 = tensor.empty() : tensor<36x1x1x64x32xf32>
    %cst_38 = arith.constant 0.000000e+00 : f32
    %pack_39 = tensor.pack %collapsed_35 padding_value(%cst_38 : f32) inner_dims_pos = [2, 1] inner_tiles = [64, 32] into %50 : tensor<36x17x64xf32> -> tensor<36x1x1x64x32xf32>
    %51 = tensor.empty() : tensor<36x1x1x32x64xf32>
    %cst_40 = arith.constant 0.000000e+00 : f32
    %pack_41 = tensor.pack %48 padding_value(%cst_40 : f32) inner_dims_pos = [1, 2] inner_tiles = [32, 64] into %51 : tensor<36x17x64xf32> -> tensor<36x1x1x32x64xf32>
    %52 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%pack_37, %pack_39 : tensor<36x1x1x32x32xf32>, tensor<36x1x1x64x32xf32>) outs(%pack_41 : tensor<36x1x1x32x64xf32>) {
    ^bb0(%in: f32, %in_45: f32, %out: f32):
      %55 = arith.mulf %in, %in_45 : f32
      %56 = arith.addf %out, %55 : f32
      linalg.yield %56 : f32
    } -> tensor<36x1x1x32x64xf32>
    %unpack_42 = tensor.unpack %52 inner_dims_pos = [1, 2] inner_tiles = [32, 64] into %48 : tensor<36x1x1x32x64xf32> -> tensor<36x17x64xf32>
    %expanded_43 = tensor.expand_shape %unpack_42 [[0, 1], [2], [3]] : tensor<36x17x64xf32> into tensor<3x12x17x64xf32>
    %53 = tensor.empty() : tensor<3x17x12x64xf32>
    %54 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_43 : tensor<3x12x17x64xf32>) outs(%53 : tensor<3x17x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x17x12x64xf32>
    %collapsed_44 = tensor.collapse_shape %54 [[0], [1], [2, 3]] : tensor<3x17x12x64xf32> into tensor<3x17x768xf32>
    return %collapsed_44 : tensor<3x17x768xf32>
  }
}
