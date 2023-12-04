#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> (d2)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map7 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> ()>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
func.func @forward(%arg0: tensor<3x17x768xf32>) -> tensor<3x17x768xf32> {
  %c0_i64 = arith.constant 0 : i64
  %cst = arith.constant dense<8.000000e+00> : tensor<f64>
  // %cst_0 = arith.constant dense_resource<__elided__> : tensor<768x768xf32>
  // %cst_1 = arith.constant dense_resource<__elided__> : tensor<768xf32>
  // %cst_2 = arith.constant dense_resource<__elided__> : tensor<768x768xf32>
  // %cst_3 = arith.constant dense_resource<__elided__> : tensor<768xf32>
  // %cst_4 = arith.constant dense_resource<__elided__> : tensor<768x768xf32>
  // %cst_5 = arith.constant dense_resource<__elided__> : tensor<768xf32>
  %cst_0 = arith.constant dense<1.> : tensor<768x768xf32>
  %cst_1 = arith.constant dense<1.> : tensor<768xf32>
  %cst_2 = arith.constant dense<1.> : tensor<768x768xf32>
  %cst_3 = arith.constant dense<1.> : tensor<768xf32>
  %cst_4 = arith.constant dense<1.> : tensor<768x768xf32>
  %cst_5 = arith.constant dense<1.> : tensor<768xf32>
  %cst_6 = arith.constant 0.000000e+00 : f32
  %cst_7 = arith.constant -3.40282347E+38 : f32
  %0 = tensor.empty() : tensor<768x768xf32>
  %1 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_4 : tensor<768x768xf32>) outs(%0 : tensor<768x768xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<768x768xf32>
  %2 = tensor.empty() : tensor<3x768x768xf32>
  %3 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%1 : tensor<768x768xf32>) outs(%2 : tensor<3x768x768xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<3x768x768xf32>
  %4 = tensor.empty() : tensor<3x17x768xf32>
  %5 = linalg.fill ins(%cst_6 : f32) outs(%4 : tensor<3x17x768xf32>) -> tensor<3x17x768xf32>
  %6 = linalg.batch_matmul ins(%arg0, %3 : tensor<3x17x768xf32>, tensor<3x768x768xf32>) outs(%5 : tensor<3x17x768xf32>) -> tensor<3x17x768xf32>
  %7 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%6, %cst_5 : tensor<3x17x768xf32>, tensor<768xf32>) outs(%4 : tensor<3x17x768xf32>) {
  ^bb0(%in: f32, %in_16: f32, %out: f32):
    %42 = arith.addf %in, %in_16 : f32
    linalg.yield %42 : f32
  } -> tensor<3x17x768xf32>
  %8 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_2 : tensor<768x768xf32>) outs(%0 : tensor<768x768xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<768x768xf32>
  %9 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%8 : tensor<768x768xf32>) outs(%2 : tensor<3x768x768xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<3x768x768xf32>
  %10 = linalg.batch_matmul ins(%arg0, %9 : tensor<3x17x768xf32>, tensor<3x768x768xf32>) outs(%5 : tensor<3x17x768xf32>) -> tensor<3x17x768xf32>
  %11 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%10, %cst_3 : tensor<3x17x768xf32>, tensor<768xf32>) outs(%4 : tensor<3x17x768xf32>) {
  ^bb0(%in: f32, %in_16: f32, %out: f32):
    %42 = arith.addf %in, %in_16 : f32
    linalg.yield %42 : f32
  } -> tensor<3x17x768xf32>
  %expanded = tensor.expand_shape %11 [[0], [1], [2, 3]] : tensor<3x17x768xf32> into tensor<3x17x12x64xf32>
  %12 = tensor.empty() : tensor<3x12x17x64xf32>
  %13 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<3x17x12x64xf32>) outs(%12 : tensor<3x12x17x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<3x12x17x64xf32>
  %14 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<768x768xf32>) outs(%0 : tensor<768x768xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<768x768xf32>
  %15 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%14 : tensor<768x768xf32>) outs(%2 : tensor<3x768x768xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<3x768x768xf32>
  %16 = linalg.batch_matmul ins(%arg0, %15 : tensor<3x17x768xf32>, tensor<3x768x768xf32>) outs(%5 : tensor<3x17x768xf32>) -> tensor<3x17x768xf32>
  %17 = linalg.generic {indexing_maps = [#map3, #map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%16, %cst_1 : tensor<3x17x768xf32>, tensor<768xf32>) outs(%4 : tensor<3x17x768xf32>) {
  ^bb0(%in: f32, %in_16: f32, %out: f32):
    %42 = arith.addf %in, %in_16 : f32
    linalg.yield %42 : f32
  } -> tensor<3x17x768xf32>
  %expanded_8 = tensor.expand_shape %17 [[0], [1], [2, 3]] : tensor<3x17x768xf32> into tensor<3x17x12x64xf32>
  %18 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_8 : tensor<3x17x12x64xf32>) outs(%12 : tensor<3x12x17x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<3x12x17x64xf32>
  %expanded_9 = tensor.expand_shape %7 [[0], [1], [2, 3]] : tensor<3x17x768xf32> into tensor<3x17x12x64xf32>
  %19 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_9 : tensor<3x17x12x64xf32>) outs(%12 : tensor<3x12x17x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<3x12x17x64xf32>
  %20 = tensor.empty() : tensor<3x12x64x17xf32>
  %21 = linalg.generic {indexing_maps = [#map5, #map7], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%13 : tensor<3x12x17x64xf32>) outs(%20 : tensor<3x12x64x17xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<3x12x64x17xf32>
  %collapsed = tensor.collapse_shape %19 [[0, 1], [2], [3]] : tensor<3x12x17x64xf32> into tensor<36x17x64xf32>
  %collapsed_10 = tensor.collapse_shape %21 [[0, 1], [2], [3]] : tensor<3x12x64x17xf32> into tensor<36x64x17xf32>
  %22 = tensor.empty() : tensor<36x17x17xf32>
  %23 = linalg.fill ins(%cst_6 : f32) outs(%22 : tensor<36x17x17xf32>) -> tensor<36x17x17xf32>
  %24 = linalg.batch_matmul ins(%collapsed, %collapsed_10 : tensor<36x17x64xf32>, tensor<36x64x17xf32>) outs(%23 : tensor<36x17x17xf32>) -> tensor<36x17x17xf32>
  %expanded_11 = tensor.expand_shape %24 [[0, 1], [2], [3]] : tensor<36x17x17xf32> into tensor<3x12x17x17xf32>
  %25 = tensor.empty() : tensor<3x12x17x17xf32>
  %26 = linalg.generic {indexing_maps = [#map5, #map8, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_11, %cst : tensor<3x12x17x17xf32>, tensor<f64>) outs(%25 : tensor<3x12x17x17xf32>) {
  ^bb0(%in: f32, %in_16: f64, %out: f32):
    %42 = arith.truncf %in_16 : f64 to f32
    %43 = arith.divf %in, %42 : f32
    linalg.yield %43 : f32
  } -> tensor<3x12x17x17xf32>
  %27 = tensor.empty() : tensor<3x12x17x1xi64>
  %28 = linalg.fill ins(%c0_i64 : i64) outs(%27 : tensor<3x12x17x1xi64>) -> tensor<3x12x17x1xi64>
  %29 = tensor.empty() : tensor<3x12x17x1xf32>
  %30 = linalg.fill ins(%cst_7 : f32) outs(%29 : tensor<3x12x17x1xf32>) -> tensor<3x12x17x1xf32>
  %31:2 = linalg.generic {indexing_maps = [#map5, #map9, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%26 : tensor<3x12x17x17xf32>) outs(%30, %28 : tensor<3x12x17x1xf32>, tensor<3x12x17x1xi64>) {
  ^bb0(%in: f32, %out: f32, %out_16: i64):
    %42 = linalg.index 3 : index
    %43 = arith.index_cast %42 : index to i64
    %44 = arith.maxf %in, %out : f32
    %45 = arith.cmpf ogt, %in, %out : f32
    %46 = arith.select %45, %43, %out_16 : i64
    linalg.yield %44, %46 : f32, i64
  } -> (tensor<3x12x17x1xf32>, tensor<3x12x17x1xi64>)
  %32 = linalg.generic {indexing_maps = [#map5, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%26, %31#0 : tensor<3x12x17x17xf32>, tensor<3x12x17x1xf32>) outs(%25 : tensor<3x12x17x17xf32>) {
  ^bb0(%in: f32, %in_16: f32, %out: f32):
    %42 = arith.subf %in, %in_16 : f32
    linalg.yield %42 : f32
  } -> tensor<3x12x17x17xf32>
  %33 = linalg.generic {indexing_maps = [#map5, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%32 : tensor<3x12x17x17xf32>) outs(%25 : tensor<3x12x17x17xf32>) {
  ^bb0(%in: f32, %out: f32):
    %42 = math.exp %in : f32
    linalg.yield %42 : f32
  } -> tensor<3x12x17x17xf32>
  %34 = linalg.fill ins(%cst_6 : f32) outs(%29 : tensor<3x12x17x1xf32>) -> tensor<3x12x17x1xf32>
  %35 = linalg.generic {indexing_maps = [#map5, #map9], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%33 : tensor<3x12x17x17xf32>) outs(%34 : tensor<3x12x17x1xf32>) {
  ^bb0(%in: f32, %out: f32):
    %42 = arith.addf %in, %out : f32
    linalg.yield %42 : f32
  } -> tensor<3x12x17x1xf32>
  %36 = linalg.generic {indexing_maps = [#map5, #map9, #map5], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%33, %35 : tensor<3x12x17x17xf32>, tensor<3x12x17x1xf32>) outs(%25 : tensor<3x12x17x17xf32>) {
  ^bb0(%in: f32, %in_16: f32, %out: f32):
    %42 = arith.divf %in, %in_16 : f32
    linalg.yield %42 : f32
  } -> tensor<3x12x17x17xf32>
  %collapsed_12 = tensor.collapse_shape %36 [[0, 1], [2], [3]] : tensor<3x12x17x17xf32> into tensor<36x17x17xf32>
  %collapsed_13 = tensor.collapse_shape %18 [[0, 1], [2], [3]] : tensor<3x12x17x64xf32> into tensor<36x17x64xf32>
  %37 = tensor.empty() : tensor<36x17x64xf32>
  %38 = linalg.fill ins(%cst_6 : f32) outs(%37 : tensor<36x17x64xf32>) -> tensor<36x17x64xf32>
  %39 = linalg.batch_matmul ins(%collapsed_12, %collapsed_13 : tensor<36x17x17xf32>, tensor<36x17x64xf32>) outs(%38 : tensor<36x17x64xf32>) -> tensor<36x17x64xf32>
  %expanded_14 = tensor.expand_shape %39 [[0, 1], [2], [3]] : tensor<36x17x64xf32> into tensor<3x12x17x64xf32>
  %40 = tensor.empty() : tensor<3x17x12x64xf32>
  %41 = linalg.generic {indexing_maps = [#map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_14 : tensor<3x12x17x64xf32>) outs(%40 : tensor<3x17x12x64xf32>) {
  ^bb0(%in: f32, %out: f32):
    linalg.yield %in : f32
  } -> tensor<3x17x12x64xf32>
  %collapsed_15 = tensor.collapse_shape %41 [[0], [1], [2, 3]] : tensor<3x17x12x64xf32> into tensor<3x17x768xf32>
  return %collapsed_15 : tensor<3x17x768xf32>
}
