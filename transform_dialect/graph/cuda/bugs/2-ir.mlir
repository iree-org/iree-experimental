
//   iree-compile --iree-hal-target-backends=cuda --iree-hal-benchmark-dispatch-repeat-count=5 transform_dialect/graph/cuda/bugs/2-ir.mlir | \
//   iree-run-module --function=forward --device=cuda --input="3x17x768xf32=1"

//      CHECK: 3x17x768xf32=[
// CHECK-SAME: [769 769 769 

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
#map5 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d3)>
#map6 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>
#map7 = affine_map<(d0, d1, d2) -> (d2)>
#map8 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map9 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map10 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map11 = affine_map<(d0, d1, d2, d3) -> ()>
module {
  func.func @forward(%arg0: tensor<3x17x768xf32>) -> tensor<3x17x768xf32> {
    %cst = arith.constant dense<8.000000e+00> : tensor<f64>
    %cst_0 = arith.constant dense<1.000000e+00> : tensor<768x768xf32>
    %cst_1 = arith.constant dense<1.000000e+00> : tensor<768xf32>
    %cst_2 = arith.constant 0.000000e+00 : f32
    %cst_3 = arith.constant -3.40282347E+38 : f32
    %c0 = arith.constant 0 : index
    %c15 = arith.constant 15 : index
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
    %padded = tensor.pad %arg0 low[%c0, %c0, %c0] high[%c0, %c15, %c0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_2 : f32
    } : tensor<3x17x768xf32> to tensor<3x32x768xf32>
    %expanded = tensor.expand_shape %padded [[0], [1, 2], [3, 4]] : tensor<3x32x768xf32> into tensor<3x1x32x1x768xf32>
    %transposed = linalg.transpose ins(%expanded : tensor<3x1x32x1x768xf32>) outs(%6 : tensor<3x1x1x32x768xf32>) permutation = [0, 1, 3, 2, 4] 
    %7 = tensor.empty() : tensor<3x1x1x768x768xf32>
    %expanded_4 = tensor.expand_shape %3 [[0], [1, 2], [3, 4]] : tensor<3x768x768xf32> into tensor<3x1x768x1x768xf32>
    %transposed_5 = linalg.transpose ins(%expanded_4 : tensor<3x1x768x1x768xf32>) outs(%7 : tensor<3x1x1x768x768xf32>) permutation = [0, 1, 3, 4, 2] 
    %8 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %padded_6 = tensor.pad %5 low[%c0, %c0, %c0] high[%c0, %c15, %c0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_2 : f32
    } : tensor<3x17x768xf32> to tensor<3x32x768xf32>
    %expanded_7 = tensor.expand_shape %padded_6 [[0], [1, 2], [3, 4]] : tensor<3x32x768xf32> into tensor<3x1x32x1x768xf32>
    %transposed_8 = linalg.transpose ins(%expanded_7 : tensor<3x1x32x1x768xf32>) outs(%8 : tensor<3x1x1x32x768xf32>) permutation = [0, 1, 3, 2, 4] 
    %collapsed = tensor.collapse_shape %transposed [[0, 1, 2], [3], [4]] : tensor<3x1x1x32x768xf32> into tensor<3x32x768xf32>
    %collapsed_9 = tensor.collapse_shape %transposed_5 [[0, 1, 2], [3], [4]] : tensor<3x1x1x768x768xf32> into tensor<3x768x768xf32>
    %collapsed_10 = tensor.collapse_shape %transposed_8 [[0, 1, 2], [3], [4]] : tensor<3x1x1x32x768xf32> into tensor<3x32x768xf32>
    %9 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed, %collapsed_9 : tensor<3x32x768xf32>, tensor<3x768x768xf32>) outs(%collapsed_10 : tensor<3x32x768xf32>) {
    ^bb0(%in: f32, %in_86: f32, %out: f32):
      %61 = arith.mulf %in, %in_86 : f32
      %62 = arith.addf %out, %61 : f32
      linalg.yield %62 : f32
    } -> tensor<3x32x768xf32>
    %expanded_11 = tensor.expand_shape %9 [[0, 1, 2], [3], [4]] : tensor<3x32x768xf32> into tensor<3x1x1x32x768xf32>
    %10 = tensor.empty() : tensor<3x1x32x1x768xf32>
    %transposed_12 = linalg.transpose ins(%expanded_11 : tensor<3x1x1x32x768xf32>) outs(%10 : tensor<3x1x32x1x768xf32>) permutation = [0, 1, 3, 2, 4] 
    %collapsed_13 = tensor.collapse_shape %transposed_12 [[0], [1, 2], [3, 4]] : tensor<3x1x32x1x768xf32> into tensor<3x32x768xf32>
    %extracted_slice = tensor.extract_slice %collapsed_13[0, 0, 0] [3, 17, 768] [1, 1, 1] : tensor<3x32x768xf32> to tensor<3x17x768xf32>
    %11 = linalg.generic {indexing_maps = [#map3, #map7, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice, %cst_1 : tensor<3x17x768xf32>, tensor<768xf32>) outs(%4 : tensor<3x17x768xf32>) {
    ^bb0(%in: f32, %in_86: f32, %out: f32):
      %61 = arith.addf %in, %in_86 : f32
      linalg.yield %61 : f32
    } -> tensor<3x17x768xf32>
    %12 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<768x768xf32>) outs(%0 : tensor<768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<768x768xf32>
    %13 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%12 : tensor<768x768xf32>) outs(%2 : tensor<3x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x768x768xf32>
    %14 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %padded_14 = tensor.pad %arg0 low[%c0, %c0, %c0] high[%c0, %c15, %c0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_2 : f32
    } : tensor<3x17x768xf32> to tensor<3x32x768xf32>
    %expanded_15 = tensor.expand_shape %padded_14 [[0], [1, 2], [3, 4]] : tensor<3x32x768xf32> into tensor<3x1x32x1x768xf32>
    %transposed_16 = linalg.transpose ins(%expanded_15 : tensor<3x1x32x1x768xf32>) outs(%14 : tensor<3x1x1x32x768xf32>) permutation = [0, 1, 3, 2, 4] 
    %15 = tensor.empty() : tensor<3x1x1x768x768xf32>
    %expanded_17 = tensor.expand_shape %13 [[0], [1, 2], [3, 4]] : tensor<3x768x768xf32> into tensor<3x1x768x1x768xf32>
    %transposed_18 = linalg.transpose ins(%expanded_17 : tensor<3x1x768x1x768xf32>) outs(%15 : tensor<3x1x1x768x768xf32>) permutation = [0, 1, 3, 4, 2] 
    %16 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %padded_19 = tensor.pad %5 low[%c0, %c0, %c0] high[%c0, %c15, %c0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_2 : f32
    } : tensor<3x17x768xf32> to tensor<3x32x768xf32>
    %expanded_20 = tensor.expand_shape %padded_19 [[0], [1, 2], [3, 4]] : tensor<3x32x768xf32> into tensor<3x1x32x1x768xf32>
    %transposed_21 = linalg.transpose ins(%expanded_20 : tensor<3x1x32x1x768xf32>) outs(%16 : tensor<3x1x1x32x768xf32>) permutation = [0, 1, 3, 2, 4] 
    %collapsed_22 = tensor.collapse_shape %transposed_16 [[0, 1, 2], [3], [4]] : tensor<3x1x1x32x768xf32> into tensor<3x32x768xf32>
    %collapsed_23 = tensor.collapse_shape %transposed_18 [[0, 1, 2], [3], [4]] : tensor<3x1x1x768x768xf32> into tensor<3x768x768xf32>
    %collapsed_24 = tensor.collapse_shape %transposed_21 [[0, 1, 2], [3], [4]] : tensor<3x1x1x32x768xf32> into tensor<3x32x768xf32>
    %17 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_22, %collapsed_23 : tensor<3x32x768xf32>, tensor<3x768x768xf32>) outs(%collapsed_24 : tensor<3x32x768xf32>) {
    ^bb0(%in: f32, %in_86: f32, %out: f32):
      %61 = arith.mulf %in, %in_86 : f32
      %62 = arith.addf %out, %61 : f32
      linalg.yield %62 : f32
    } -> tensor<3x32x768xf32>
    %expanded_25 = tensor.expand_shape %17 [[0, 1, 2], [3], [4]] : tensor<3x32x768xf32> into tensor<3x1x1x32x768xf32>
    %18 = tensor.empty() : tensor<3x1x32x1x768xf32>
    %transposed_26 = linalg.transpose ins(%expanded_25 : tensor<3x1x1x32x768xf32>) outs(%18 : tensor<3x1x32x1x768xf32>) permutation = [0, 1, 3, 2, 4] 
    %collapsed_27 = tensor.collapse_shape %transposed_26 [[0], [1, 2], [3, 4]] : tensor<3x1x32x1x768xf32> into tensor<3x32x768xf32>
    %extracted_slice_28 = tensor.extract_slice %collapsed_27[0, 0, 0] [3, 17, 768] [1, 1, 1] : tensor<3x32x768xf32> to tensor<3x17x768xf32>
    %19 = linalg.generic {indexing_maps = [#map3, #map7, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_28, %cst_1 : tensor<3x17x768xf32>, tensor<768xf32>) outs(%4 : tensor<3x17x768xf32>) {
    ^bb0(%in: f32, %in_86: f32, %out: f32):
      %61 = arith.addf %in, %in_86 : f32
      linalg.yield %61 : f32
    } -> tensor<3x17x768xf32>
    %expanded_29 = tensor.expand_shape %19 [[0], [1], [2, 3]] : tensor<3x17x768xf32> into tensor<3x17x12x64xf32>
    %20 = tensor.empty() : tensor<3x12x17x64xf32>
    %21 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_29 : tensor<3x17x12x64xf32>) outs(%20 : tensor<3x12x17x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x12x17x64xf32>
    %22 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<768x768xf32>) outs(%0 : tensor<768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<768x768xf32>
    %23 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%22 : tensor<768x768xf32>) outs(%2 : tensor<3x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x768x768xf32>
    %24 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %padded_30 = tensor.pad %arg0 low[%c0, %c0, %c0] high[%c0, %c15, %c0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_2 : f32
    } : tensor<3x17x768xf32> to tensor<3x32x768xf32>
    %expanded_31 = tensor.expand_shape %padded_30 [[0], [1, 2], [3, 4]] : tensor<3x32x768xf32> into tensor<3x1x32x1x768xf32>
    %transposed_32 = linalg.transpose ins(%expanded_31 : tensor<3x1x32x1x768xf32>) outs(%24 : tensor<3x1x1x32x768xf32>) permutation = [0, 1, 3, 2, 4] 
    %25 = tensor.empty() : tensor<3x1x1x768x768xf32>
    %expanded_33 = tensor.expand_shape %23 [[0], [1, 2], [3, 4]] : tensor<3x768x768xf32> into tensor<3x1x768x1x768xf32>
    %transposed_34 = linalg.transpose ins(%expanded_33 : tensor<3x1x768x1x768xf32>) outs(%25 : tensor<3x1x1x768x768xf32>) permutation = [0, 1, 3, 4, 2] 
    %26 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %padded_35 = tensor.pad %5 low[%c0, %c0, %c0] high[%c0, %c15, %c0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_2 : f32
    } : tensor<3x17x768xf32> to tensor<3x32x768xf32>
    %expanded_36 = tensor.expand_shape %padded_35 [[0], [1, 2], [3, 4]] : tensor<3x32x768xf32> into tensor<3x1x32x1x768xf32>
    %transposed_37 = linalg.transpose ins(%expanded_36 : tensor<3x1x32x1x768xf32>) outs(%26 : tensor<3x1x1x32x768xf32>) permutation = [0, 1, 3, 2, 4] 
    %collapsed_38 = tensor.collapse_shape %transposed_32 [[0, 1, 2], [3], [4]] : tensor<3x1x1x32x768xf32> into tensor<3x32x768xf32>
    %collapsed_39 = tensor.collapse_shape %transposed_34 [[0, 1, 2], [3], [4]] : tensor<3x1x1x768x768xf32> into tensor<3x768x768xf32>
    %collapsed_40 = tensor.collapse_shape %transposed_37 [[0, 1, 2], [3], [4]] : tensor<3x1x1x32x768xf32> into tensor<3x32x768xf32>
    %27 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_38, %collapsed_39 : tensor<3x32x768xf32>, tensor<3x768x768xf32>) outs(%collapsed_40 : tensor<3x32x768xf32>) {
    ^bb0(%in: f32, %in_86: f32, %out: f32):
      %61 = arith.mulf %in, %in_86 : f32
      %62 = arith.addf %out, %61 : f32
      linalg.yield %62 : f32
    } -> tensor<3x32x768xf32>
    %expanded_41 = tensor.expand_shape %27 [[0, 1, 2], [3], [4]] : tensor<3x32x768xf32> into tensor<3x1x1x32x768xf32>
    %28 = tensor.empty() : tensor<3x1x32x1x768xf32>
    %transposed_42 = linalg.transpose ins(%expanded_41 : tensor<3x1x1x32x768xf32>) outs(%28 : tensor<3x1x32x1x768xf32>) permutation = [0, 1, 3, 2, 4] 
    %collapsed_43 = tensor.collapse_shape %transposed_42 [[0], [1, 2], [3, 4]] : tensor<3x1x32x1x768xf32> into tensor<3x32x768xf32>
    %extracted_slice_44 = tensor.extract_slice %collapsed_43[0, 0, 0] [3, 17, 768] [1, 1, 1] : tensor<3x32x768xf32> to tensor<3x17x768xf32>
    %29 = linalg.generic {indexing_maps = [#map3, #map7, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_44, %cst_1 : tensor<3x17x768xf32>, tensor<768xf32>) outs(%4 : tensor<3x17x768xf32>) {
    ^bb0(%in: f32, %in_86: f32, %out: f32):
      %61 = arith.addf %in, %in_86 : f32
      linalg.yield %61 : f32
    } -> tensor<3x17x768xf32>
    %expanded_45 = tensor.expand_shape %29 [[0], [1], [2, 3]] : tensor<3x17x768xf32> into tensor<3x17x12x64xf32>
    %30 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_45 : tensor<3x17x12x64xf32>) outs(%20 : tensor<3x12x17x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x12x17x64xf32>
    %expanded_46 = tensor.expand_shape %11 [[0], [1], [2, 3]] : tensor<3x17x768xf32> into tensor<3x17x12x64xf32>
    %31 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_46 : tensor<3x17x12x64xf32>) outs(%20 : tensor<3x12x17x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x12x17x64xf32>
    %32 = tensor.empty() : tensor<3x12x64x17xf32>
    %33 = linalg.generic {indexing_maps = [#map8, #map10], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%21 : tensor<3x12x17x64xf32>) outs(%32 : tensor<3x12x64x17xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x12x64x17xf32>
    %collapsed_47 = tensor.collapse_shape %31 [[0, 1], [2], [3]] : tensor<3x12x17x64xf32> into tensor<36x17x64xf32>
    %collapsed_48 = tensor.collapse_shape %33 [[0, 1], [2], [3]] : tensor<3x12x64x17xf32> into tensor<36x64x17xf32>
    %34 = tensor.empty() : tensor<36x17x17xf32>
    %35 = linalg.fill ins(%cst_2 : f32) outs(%34 : tensor<36x17x17xf32>) -> tensor<36x17x17xf32>
    %36 = tensor.empty() : tensor<36x1x1x32x64xf32>
    %padded_49 = tensor.pad %collapsed_47 low[%c0, %c0, %c0] high[%c0, %c15, %c0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_2 : f32
    } : tensor<36x17x64xf32> to tensor<36x32x64xf32>
    %expanded_50 = tensor.expand_shape %padded_49 [[0], [1, 2], [3, 4]] : tensor<36x32x64xf32> into tensor<36x1x32x1x64xf32>
    %transposed_51 = linalg.transpose ins(%expanded_50 : tensor<36x1x32x1x64xf32>) outs(%36 : tensor<36x1x1x32x64xf32>) permutation = [0, 1, 3, 2, 4] 
    %37 = tensor.empty() : tensor<36x1x1x32x64xf32>
    %padded_52 = tensor.pad %collapsed_48 low[%c0, %c0, %c0] high[%c0, %c0, %c15] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_2 : f32
    } : tensor<36x64x17xf32> to tensor<36x64x32xf32>
    %expanded_53 = tensor.expand_shape %padded_52 [[0], [1, 2], [3, 4]] : tensor<36x64x32xf32> into tensor<36x1x64x1x32xf32>
    %transposed_54 = linalg.transpose ins(%expanded_53 : tensor<36x1x64x1x32xf32>) outs(%37 : tensor<36x1x1x32x64xf32>) permutation = [0, 1, 3, 4, 2] 
    %38 = tensor.empty() : tensor<36x1x1x32x32xf32>
    %padded_55 = tensor.pad %35 low[%c0, %c0, %c0] high[%c0, %c15, %c15] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_2 : f32
    } : tensor<36x17x17xf32> to tensor<36x32x32xf32>
    %expanded_56 = tensor.expand_shape %padded_55 [[0], [1, 2], [3, 4]] : tensor<36x32x32xf32> into tensor<36x1x32x1x32xf32>
    %transposed_57 = linalg.transpose ins(%expanded_56 : tensor<36x1x32x1x32xf32>) outs(%38 : tensor<36x1x1x32x32xf32>) permutation = [0, 1, 3, 2, 4] 
    %collapsed_58 = tensor.collapse_shape %transposed_51 [[0, 1, 2], [3], [4]] : tensor<36x1x1x32x64xf32> into tensor<36x32x64xf32>
    %collapsed_59 = tensor.collapse_shape %transposed_54 [[0, 1, 2], [3], [4]] : tensor<36x1x1x32x64xf32> into tensor<36x32x64xf32>
    %collapsed_60 = tensor.collapse_shape %transposed_57 [[0, 1, 2], [3], [4]] : tensor<36x1x1x32x32xf32> into tensor<36x32x32xf32>
    %39 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_58, %collapsed_59 : tensor<36x32x64xf32>, tensor<36x32x64xf32>) outs(%collapsed_60 : tensor<36x32x32xf32>) {
    ^bb0(%in: f32, %in_86: f32, %out: f32):
      %61 = arith.mulf %in, %in_86 : f32
      %62 = arith.addf %out, %61 : f32
      linalg.yield %62 : f32
    } -> tensor<36x32x32xf32>
    %expanded_61 = tensor.expand_shape %39 [[0, 1, 2], [3], [4]] : tensor<36x32x32xf32> into tensor<36x1x1x32x32xf32>
    %40 = tensor.empty() : tensor<36x1x32x1x32xf32>
    %transposed_62 = linalg.transpose ins(%expanded_61 : tensor<36x1x1x32x32xf32>) outs(%40 : tensor<36x1x32x1x32xf32>) permutation = [0, 1, 3, 2, 4] 
    %collapsed_63 = tensor.collapse_shape %transposed_62 [[0], [1, 2], [3, 4]] : tensor<36x1x32x1x32xf32> into tensor<36x32x32xf32>
    %extracted_slice_64 = tensor.extract_slice %collapsed_63[0, 0, 0] [36, 17, 17] [1, 1, 1] : tensor<36x32x32xf32> to tensor<36x17x17xf32>
    %expanded_65 = tensor.expand_shape %extracted_slice_64 [[0, 1], [2], [3]] : tensor<36x17x17xf32> into tensor<3x12x17x17xf32>
    %41 = tensor.empty() : tensor<3x12x17x17xf32>
    %42 = linalg.generic {indexing_maps = [#map8, #map11, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_65, %cst : tensor<3x12x17x17xf32>, tensor<f64>) outs(%41 : tensor<3x12x17x17xf32>) {
    ^bb0(%in: f32, %in_86: f64, %out: f32):
      %61 = arith.truncf %in_86 : f64 to f32
      %62 = arith.divf %in, %61 : f32
      linalg.yield %62 : f32
    } -> tensor<3x12x17x17xf32>
    %43 = tensor.empty() : tensor<3x12x17xf32>
    %44 = linalg.fill ins(%cst_3 : f32) outs(%43 : tensor<3x12x17xf32>) -> tensor<3x12x17xf32>
    %45 = linalg.generic {indexing_maps = [#map8, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%42 : tensor<3x12x17x17xf32>) outs(%44 : tensor<3x12x17xf32>) {
    ^bb0(%in: f32, %out: f32):
      %61 = arith.maxf %in, %out : f32
      linalg.yield %61 : f32
    } -> tensor<3x12x17xf32>
    %46 = linalg.generic {indexing_maps = [#map8, #map6, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%42, %45 : tensor<3x12x17x17xf32>, tensor<3x12x17xf32>) outs(%41 : tensor<3x12x17x17xf32>) {
    ^bb0(%in: f32, %in_86: f32, %out: f32):
      %61 = arith.subf %in, %in_86 : f32
      linalg.yield %61 : f32
    } -> tensor<3x12x17x17xf32>
    %47 = linalg.generic {indexing_maps = [#map8, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%46 : tensor<3x12x17x17xf32>) outs(%41 : tensor<3x12x17x17xf32>) {
    ^bb0(%in: f32, %out: f32):
      %61 = math.exp %in : f32
      linalg.yield %61 : f32
    } -> tensor<3x12x17x17xf32>
    %48 = tensor.empty() : tensor<3x12x17xf32>
    %49 = linalg.fill ins(%cst_2 : f32) outs(%48 : tensor<3x12x17xf32>) -> tensor<3x12x17xf32>
    %50 = linalg.generic {indexing_maps = [#map8, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%47 : tensor<3x12x17x17xf32>) outs(%49 : tensor<3x12x17xf32>) {
    ^bb0(%in: f32, %out: f32):
      %61 = arith.addf %in, %out : f32
      linalg.yield %61 : f32
    } -> tensor<3x12x17xf32>
    %51 = linalg.generic {indexing_maps = [#map8, #map6, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%47, %50 : tensor<3x12x17x17xf32>, tensor<3x12x17xf32>) outs(%41 : tensor<3x12x17x17xf32>) {
    ^bb0(%in: f32, %in_86: f32, %out: f32):
      %61 = arith.divf %in, %in_86 : f32
      linalg.yield %61 : f32
    } -> tensor<3x12x17x17xf32>
    %collapsed_66 = tensor.collapse_shape %51 [[0, 1], [2], [3]] : tensor<3x12x17x17xf32> into tensor<36x17x17xf32>
    %collapsed_67 = tensor.collapse_shape %30 [[0, 1], [2], [3]] : tensor<3x12x17x64xf32> into tensor<36x17x64xf32>
    %52 = tensor.empty() : tensor<36x17x64xf32>
    %53 = linalg.fill ins(%cst_2 : f32) outs(%52 : tensor<36x17x64xf32>) -> tensor<36x17x64xf32>
    %54 = tensor.empty() : tensor<36x1x1x32x32xf32>
    %padded_68 = tensor.pad %collapsed_66 low[%c0, %c0, %c0] high[%c0, %c15, %c15] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_2 : f32
    } : tensor<36x17x17xf32> to tensor<36x32x32xf32>
    %expanded_69 = tensor.expand_shape %padded_68 [[0], [1, 2], [3, 4]] : tensor<36x32x32xf32> into tensor<36x1x32x1x32xf32>
    %transposed_70 = linalg.transpose ins(%expanded_69 : tensor<36x1x32x1x32xf32>) outs(%54 : tensor<36x1x1x32x32xf32>) permutation = [0, 1, 3, 2, 4] 
    %55 = tensor.empty() : tensor<36x1x1x64x32xf32>
    %padded_71 = tensor.pad %collapsed_67 low[%c0, %c0, %c0] high[%c0, %c15, %c0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_2 : f32
    } : tensor<36x17x64xf32> to tensor<36x32x64xf32>
    %expanded_72 = tensor.expand_shape %padded_71 [[0], [1, 2], [3, 4]] : tensor<36x32x64xf32> into tensor<36x1x32x1x64xf32>
    %transposed_73 = linalg.transpose ins(%expanded_72 : tensor<36x1x32x1x64xf32>) outs(%55 : tensor<36x1x1x64x32xf32>) permutation = [0, 1, 3, 4, 2] 
    %56 = tensor.empty() : tensor<36x1x1x32x64xf32>
    %padded_74 = tensor.pad %53 low[%c0, %c0, %c0] high[%c0, %c15, %c0] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_2 : f32
    } : tensor<36x17x64xf32> to tensor<36x32x64xf32>
    %expanded_75 = tensor.expand_shape %padded_74 [[0], [1, 2], [3, 4]] : tensor<36x32x64xf32> into tensor<36x1x32x1x64xf32>
    %transposed_76 = linalg.transpose ins(%expanded_75 : tensor<36x1x32x1x64xf32>) outs(%56 : tensor<36x1x1x32x64xf32>) permutation = [0, 1, 3, 2, 4] 
    %collapsed_77 = tensor.collapse_shape %transposed_70 [[0, 1, 2], [3], [4]] : tensor<36x1x1x32x32xf32> into tensor<36x32x32xf32>
    %collapsed_78 = tensor.collapse_shape %transposed_73 [[0, 1, 2], [3], [4]] : tensor<36x1x1x64x32xf32> into tensor<36x64x32xf32>
    %collapsed_79 = tensor.collapse_shape %transposed_76 [[0, 1, 2], [3], [4]] : tensor<36x1x1x32x64xf32> into tensor<36x32x64xf32>
    %57 = linalg.generic {indexing_maps = [#map4, #map5, #map6], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%collapsed_77, %collapsed_78 : tensor<36x32x32xf32>, tensor<36x64x32xf32>) outs(%collapsed_79 : tensor<36x32x64xf32>) {
    ^bb0(%in: f32, %in_86: f32, %out: f32):
      %61 = arith.mulf %in, %in_86 : f32
      %62 = arith.addf %out, %61 : f32
      linalg.yield %62 : f32
    } -> tensor<36x32x64xf32>
    %expanded_80 = tensor.expand_shape %57 [[0, 1, 2], [3], [4]] : tensor<36x32x64xf32> into tensor<36x1x1x32x64xf32>
    %58 = tensor.empty() : tensor<36x1x32x1x64xf32>
    %transposed_81 = linalg.transpose ins(%expanded_80 : tensor<36x1x1x32x64xf32>) outs(%58 : tensor<36x1x32x1x64xf32>) permutation = [0, 1, 3, 2, 4] 
    %collapsed_82 = tensor.collapse_shape %transposed_81 [[0], [1, 2], [3, 4]] : tensor<36x1x32x1x64xf32> into tensor<36x32x64xf32>
    %extracted_slice_83 = tensor.extract_slice %collapsed_82[0, 0, 0] [36, 17, 64] [1, 1, 1] : tensor<36x32x64xf32> to tensor<36x17x64xf32>
    %expanded_84 = tensor.expand_shape %extracted_slice_83 [[0, 1], [2], [3]] : tensor<36x17x64xf32> into tensor<3x12x17x64xf32>
    %59 = tensor.empty() : tensor<3x17x12x64xf32>
    %60 = linalg.generic {indexing_maps = [#map8, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_84 : tensor<3x12x17x64xf32>) outs(%59 : tensor<3x17x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x17x12x64xf32>
    %collapsed_85 = tensor.collapse_shape %60 [[0], [1], [2, 3]] : tensor<3x17x12x64xf32> into tensor<3x17x768xf32>
    return %collapsed_85 : tensor<3x17x768xf32>
  }
}

