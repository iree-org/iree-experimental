
//   iree-compile --iree-hal-target-backends=cuda --iree-hal-benchmark-dispatch-repeat-count=5 transform_dialect/graph/cuda/bugs/3-ir.mlir | \
//   iree-run-module --function=forward --device=cuda --input="3x17x768xf32=1"

//      CHECK: 3x17x768xf32=[
// CHECK-SAME: [769 769 769 

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d1, d0)>
#map2 = affine_map<(d0, d1, d2) -> (d1, d2)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map4 = affine_map<(d0, d1, d2) -> ()>
#map5 = affine_map<() -> (0)>
#map6 = affine_map<() -> (15)>
#map7 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#map8 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d2, d4)>
#map9 = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3, d4, d2)>
#map10 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d3, d4, d6)>
#map11 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d3, d2, d5, d6)>
#map12 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d4, d5)>
#map13 = affine_map<(d0, d1, d2) -> (d2)>
#map14 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map15 = affine_map<(d0, d1, d2, d3) -> (d0, d2, d1, d3)>
#map16 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>
#map17 = affine_map<(d0, d1, d2, d3) -> ()>
#map18 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, 0)>
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
    %5 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_2 : f32) outs(%4 : tensor<3x17x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x17x768xf32>
    %6 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %cst_4 = arith.constant 0.000000e+00 : f32
    %c0 = arith.constant 0 : index
    %c0_5 = arith.constant 0 : index
    %c3 = arith.constant 3 : index
    %7 = affine.apply #map5()
    %c1 = arith.constant 1 : index
    %c17 = arith.constant 17 : index
    %8 = affine.apply #map6()
    %c2 = arith.constant 2 : index
    %c768 = arith.constant 768 : index
    %9 = affine.apply #map5()
    %padded = tensor.pad %arg0 low[%c0, %c0, %c0] high[%7, %8, %9] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_4 : f32
    } : tensor<3x17x768xf32> to tensor<3x32x768xf32>
    %expanded = tensor.expand_shape %padded [[0], [1, 2], [3, 4]] : tensor<3x32x768xf32> into tensor<3x1x32x1x768xf32>
    %10 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded : tensor<3x1x32x1x768xf32>) outs(%6 : tensor<3x1x1x32x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x1x1x32x768xf32>
    %11 = tensor.empty() : tensor<3x1x1x768x768xf32>
    %cst_6 = arith.constant 0.000000e+00 : f32
    %c0_7 = arith.constant 0 : index
    %c0_8 = arith.constant 0 : index
    %c3_9 = arith.constant 3 : index
    %12 = affine.apply #map5()
    %c1_10 = arith.constant 1 : index
    %c768_11 = arith.constant 768 : index
    %13 = affine.apply #map5()
    %c2_12 = arith.constant 2 : index
    %c768_13 = arith.constant 768 : index
    %14 = affine.apply #map5()
    %padded_14 = tensor.pad %3 low[%c0_7, %c0_7, %c0_7] high[%12, %13, %14] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_6 : f32
    } : tensor<3x768x768xf32> to tensor<3x768x768xf32>
    %expanded_15 = tensor.expand_shape %padded_14 [[0], [1, 2], [3, 4]] : tensor<3x768x768xf32> into tensor<3x1x768x1x768xf32>
    %15 = linalg.generic {indexing_maps = [#map7, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_15 : tensor<3x1x768x1x768xf32>) outs(%11 : tensor<3x1x1x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x1x1x768x768xf32>
    %16 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %cst_16 = arith.constant 0.000000e+00 : f32
    %c0_17 = arith.constant 0 : index
    %c0_18 = arith.constant 0 : index
    %c3_19 = arith.constant 3 : index
    %17 = affine.apply #map5()
    %c1_20 = arith.constant 1 : index
    %c17_21 = arith.constant 17 : index
    %18 = affine.apply #map6()
    %c2_22 = arith.constant 2 : index
    %c768_23 = arith.constant 768 : index
    %19 = affine.apply #map5()
    %padded_24 = tensor.pad %5 low[%c0_17, %c0_17, %c0_17] high[%17, %18, %19] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_16 : f32
    } : tensor<3x17x768xf32> to tensor<3x32x768xf32>
    %expanded_25 = tensor.expand_shape %padded_24 [[0], [1, 2], [3, 4]] : tensor<3x32x768xf32> into tensor<3x1x32x1x768xf32>
    %20 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_25 : tensor<3x1x32x1x768xf32>) outs(%16 : tensor<3x1x1x32x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x1x1x32x768xf32>
    %21 = linalg.generic {indexing_maps = [#map10, #map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%10, %15 : tensor<3x1x1x32x768xf32>, tensor<3x1x1x768x768xf32>) outs(%20 : tensor<3x1x1x32x768xf32>) {
    ^bb0(%in: f32, %in_162: f32, %out: f32):
      %125 = arith.mulf %in, %in_162 : f32
      %126 = arith.addf %out, %125 : f32
      linalg.yield %126 : f32
    } -> tensor<3x1x1x32x768xf32>
    %22 = tensor.empty() : tensor<3x1x32x1x768xf32>
    %23 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%21 : tensor<3x1x1x32x768xf32>) outs(%22 : tensor<3x1x32x1x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x1x32x1x768xf32>
    %collapsed = tensor.collapse_shape %23 [[0], [1, 2], [3, 4]] : tensor<3x1x32x1x768xf32> into tensor<3x32x768xf32>
    %extracted_slice = tensor.extract_slice %collapsed[0, 0, 0] [3, 17, 768] [1, 1, 1] : tensor<3x32x768xf32> to tensor<3x17x768xf32>
    %24 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice, %cst_1 : tensor<3x17x768xf32>, tensor<768xf32>) outs(%4 : tensor<3x17x768xf32>) {
    ^bb0(%in: f32, %in_162: f32, %out: f32):
      %125 = arith.addf %in, %in_162 : f32
      linalg.yield %125 : f32
    } -> tensor<3x17x768xf32>
    %25 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<768x768xf32>) outs(%0 : tensor<768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<768x768xf32>
    %26 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%25 : tensor<768x768xf32>) outs(%2 : tensor<3x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x768x768xf32>
    %27 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %cst_26 = arith.constant 0.000000e+00 : f32
    %c0_27 = arith.constant 0 : index
    %c0_28 = arith.constant 0 : index
    %c3_29 = arith.constant 3 : index
    %28 = affine.apply #map5()
    %c1_30 = arith.constant 1 : index
    %c17_31 = arith.constant 17 : index
    %29 = affine.apply #map6()
    %c2_32 = arith.constant 2 : index
    %c768_33 = arith.constant 768 : index
    %30 = affine.apply #map5()
    %padded_34 = tensor.pad %arg0 low[%c0_27, %c0_27, %c0_27] high[%28, %29, %30] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_26 : f32
    } : tensor<3x17x768xf32> to tensor<3x32x768xf32>
    %expanded_35 = tensor.expand_shape %padded_34 [[0], [1, 2], [3, 4]] : tensor<3x32x768xf32> into tensor<3x1x32x1x768xf32>
    %31 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_35 : tensor<3x1x32x1x768xf32>) outs(%27 : tensor<3x1x1x32x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x1x1x32x768xf32>
    %32 = tensor.empty() : tensor<3x1x1x768x768xf32>
    %cst_36 = arith.constant 0.000000e+00 : f32
    %c0_37 = arith.constant 0 : index
    %c0_38 = arith.constant 0 : index
    %c3_39 = arith.constant 3 : index
    %33 = affine.apply #map5()
    %c1_40 = arith.constant 1 : index
    %c768_41 = arith.constant 768 : index
    %34 = affine.apply #map5()
    %c2_42 = arith.constant 2 : index
    %c768_43 = arith.constant 768 : index
    %35 = affine.apply #map5()
    %padded_44 = tensor.pad %26 low[%c0_37, %c0_37, %c0_37] high[%33, %34, %35] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_36 : f32
    } : tensor<3x768x768xf32> to tensor<3x768x768xf32>
    %expanded_45 = tensor.expand_shape %padded_44 [[0], [1, 2], [3, 4]] : tensor<3x768x768xf32> into tensor<3x1x768x1x768xf32>
    %36 = linalg.generic {indexing_maps = [#map7, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_45 : tensor<3x1x768x1x768xf32>) outs(%32 : tensor<3x1x1x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x1x1x768x768xf32>
    %37 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %cst_46 = arith.constant 0.000000e+00 : f32
    %c0_47 = arith.constant 0 : index
    %c0_48 = arith.constant 0 : index
    %c3_49 = arith.constant 3 : index
    %38 = affine.apply #map5()
    %c1_50 = arith.constant 1 : index
    %c17_51 = arith.constant 17 : index
    %39 = affine.apply #map6()
    %c2_52 = arith.constant 2 : index
    %c768_53 = arith.constant 768 : index
    %40 = affine.apply #map5()
    %padded_54 = tensor.pad %5 low[%c0_47, %c0_47, %c0_47] high[%38, %39, %40] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_46 : f32
    } : tensor<3x17x768xf32> to tensor<3x32x768xf32>
    %expanded_55 = tensor.expand_shape %padded_54 [[0], [1, 2], [3, 4]] : tensor<3x32x768xf32> into tensor<3x1x32x1x768xf32>
    %41 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_55 : tensor<3x1x32x1x768xf32>) outs(%37 : tensor<3x1x1x32x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x1x1x32x768xf32>
    %42 = linalg.generic {indexing_maps = [#map10, #map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%31, %36 : tensor<3x1x1x32x768xf32>, tensor<3x1x1x768x768xf32>) outs(%41 : tensor<3x1x1x32x768xf32>) {
    ^bb0(%in: f32, %in_162: f32, %out: f32):
      %125 = arith.mulf %in, %in_162 : f32
      %126 = arith.addf %out, %125 : f32
      linalg.yield %126 : f32
    } -> tensor<3x1x1x32x768xf32>
    %43 = tensor.empty() : tensor<3x1x32x1x768xf32>
    %44 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%42 : tensor<3x1x1x32x768xf32>) outs(%43 : tensor<3x1x32x1x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x1x32x1x768xf32>
    %collapsed_56 = tensor.collapse_shape %44 [[0], [1, 2], [3, 4]] : tensor<3x1x32x1x768xf32> into tensor<3x32x768xf32>
    %extracted_slice_57 = tensor.extract_slice %collapsed_56[0, 0, 0] [3, 17, 768] [1, 1, 1] : tensor<3x32x768xf32> to tensor<3x17x768xf32>
    %45 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_57, %cst_1 : tensor<3x17x768xf32>, tensor<768xf32>) outs(%4 : tensor<3x17x768xf32>) {
    ^bb0(%in: f32, %in_162: f32, %out: f32):
      %125 = arith.addf %in, %in_162 : f32
      linalg.yield %125 : f32
    } -> tensor<3x17x768xf32>
    %expanded_58 = tensor.expand_shape %45 [[0], [1], [2, 3]] : tensor<3x17x768xf32> into tensor<3x17x12x64xf32>
    %46 = tensor.empty() : tensor<3x12x17x64xf32>
    %47 = linalg.generic {indexing_maps = [#map14, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_58 : tensor<3x17x12x64xf32>) outs(%46 : tensor<3x12x17x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x12x17x64xf32>
    %48 = linalg.generic {indexing_maps = [#map, #map1], iterator_types = ["parallel", "parallel"]} ins(%cst_0 : tensor<768x768xf32>) outs(%0 : tensor<768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<768x768xf32>
    %49 = linalg.generic {indexing_maps = [#map2, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%48 : tensor<768x768xf32>) outs(%2 : tensor<3x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x768x768xf32>
    %50 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %cst_59 = arith.constant 0.000000e+00 : f32
    %c0_60 = arith.constant 0 : index
    %c0_61 = arith.constant 0 : index
    %c3_62 = arith.constant 3 : index
    %51 = affine.apply #map5()
    %c1_63 = arith.constant 1 : index
    %c17_64 = arith.constant 17 : index
    %52 = affine.apply #map6()
    %c2_65 = arith.constant 2 : index
    %c768_66 = arith.constant 768 : index
    %53 = affine.apply #map5()
    %padded_67 = tensor.pad %arg0 low[%c0_60, %c0_60, %c0_60] high[%51, %52, %53] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_59 : f32
    } : tensor<3x17x768xf32> to tensor<3x32x768xf32>
    %expanded_68 = tensor.expand_shape %padded_67 [[0], [1, 2], [3, 4]] : tensor<3x32x768xf32> into tensor<3x1x32x1x768xf32>
    %54 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_68 : tensor<3x1x32x1x768xf32>) outs(%50 : tensor<3x1x1x32x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x1x1x32x768xf32>
    %55 = tensor.empty() : tensor<3x1x1x768x768xf32>
    %cst_69 = arith.constant 0.000000e+00 : f32
    %c0_70 = arith.constant 0 : index
    %c0_71 = arith.constant 0 : index
    %c3_72 = arith.constant 3 : index
    %56 = affine.apply #map5()
    %c1_73 = arith.constant 1 : index
    %c768_74 = arith.constant 768 : index
    %57 = affine.apply #map5()
    %c2_75 = arith.constant 2 : index
    %c768_76 = arith.constant 768 : index
    %58 = affine.apply #map5()
    %padded_77 = tensor.pad %49 low[%c0_70, %c0_70, %c0_70] high[%56, %57, %58] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_69 : f32
    } : tensor<3x768x768xf32> to tensor<3x768x768xf32>
    %expanded_78 = tensor.expand_shape %padded_77 [[0], [1, 2], [3, 4]] : tensor<3x768x768xf32> into tensor<3x1x768x1x768xf32>
    %59 = linalg.generic {indexing_maps = [#map7, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_78 : tensor<3x1x768x1x768xf32>) outs(%55 : tensor<3x1x1x768x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x1x1x768x768xf32>
    %60 = tensor.empty() : tensor<3x1x1x32x768xf32>
    %cst_79 = arith.constant 0.000000e+00 : f32
    %c0_80 = arith.constant 0 : index
    %c0_81 = arith.constant 0 : index
    %c3_82 = arith.constant 3 : index
    %61 = affine.apply #map5()
    %c1_83 = arith.constant 1 : index
    %c17_84 = arith.constant 17 : index
    %62 = affine.apply #map6()
    %c2_85 = arith.constant 2 : index
    %c768_86 = arith.constant 768 : index
    %63 = affine.apply #map5()
    %padded_87 = tensor.pad %5 low[%c0_80, %c0_80, %c0_80] high[%61, %62, %63] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_79 : f32
    } : tensor<3x17x768xf32> to tensor<3x32x768xf32>
    %expanded_88 = tensor.expand_shape %padded_87 [[0], [1, 2], [3, 4]] : tensor<3x32x768xf32> into tensor<3x1x32x1x768xf32>
    %64 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_88 : tensor<3x1x32x1x768xf32>) outs(%60 : tensor<3x1x1x32x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x1x1x32x768xf32>
    %65 = linalg.generic {indexing_maps = [#map10, #map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%54, %59 : tensor<3x1x1x32x768xf32>, tensor<3x1x1x768x768xf32>) outs(%64 : tensor<3x1x1x32x768xf32>) {
    ^bb0(%in: f32, %in_162: f32, %out: f32):
      %125 = arith.mulf %in, %in_162 : f32
      %126 = arith.addf %out, %125 : f32
      linalg.yield %126 : f32
    } -> tensor<3x1x1x32x768xf32>
    %66 = tensor.empty() : tensor<3x1x32x1x768xf32>
    %67 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%65 : tensor<3x1x1x32x768xf32>) outs(%66 : tensor<3x1x32x1x768xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x1x32x1x768xf32>
    %collapsed_89 = tensor.collapse_shape %67 [[0], [1, 2], [3, 4]] : tensor<3x1x32x1x768xf32> into tensor<3x32x768xf32>
    %extracted_slice_90 = tensor.extract_slice %collapsed_89[0, 0, 0] [3, 17, 768] [1, 1, 1] : tensor<3x32x768xf32> to tensor<3x17x768xf32>
    %68 = linalg.generic {indexing_maps = [#map3, #map13, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%extracted_slice_90, %cst_1 : tensor<3x17x768xf32>, tensor<768xf32>) outs(%4 : tensor<3x17x768xf32>) {
    ^bb0(%in: f32, %in_162: f32, %out: f32):
      %125 = arith.addf %in, %in_162 : f32
      linalg.yield %125 : f32
    } -> tensor<3x17x768xf32>
    %expanded_91 = tensor.expand_shape %68 [[0], [1], [2, 3]] : tensor<3x17x768xf32> into tensor<3x17x12x64xf32>
    %69 = linalg.generic {indexing_maps = [#map14, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_91 : tensor<3x17x12x64xf32>) outs(%46 : tensor<3x12x17x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x12x17x64xf32>
    %expanded_92 = tensor.expand_shape %24 [[0], [1], [2, 3]] : tensor<3x17x768xf32> into tensor<3x17x12x64xf32>
    %70 = linalg.generic {indexing_maps = [#map14, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_92 : tensor<3x17x12x64xf32>) outs(%46 : tensor<3x12x17x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x12x17x64xf32>
    %71 = tensor.empty() : tensor<3x12x64x17xf32>
    %72 = linalg.generic {indexing_maps = [#map14, #map16], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%47 : tensor<3x12x17x64xf32>) outs(%71 : tensor<3x12x64x17xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x12x64x17xf32>
    %collapsed_93 = tensor.collapse_shape %70 [[0, 1], [2], [3]] : tensor<3x12x17x64xf32> into tensor<36x17x64xf32>
    %collapsed_94 = tensor.collapse_shape %72 [[0, 1], [2], [3]] : tensor<3x12x64x17xf32> into tensor<36x64x17xf32>
    %73 = tensor.empty() : tensor<36x17x17xf32>
    %74 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_2 : f32) outs(%73 : tensor<36x17x17xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<36x17x17xf32>
    %75 = tensor.empty() : tensor<36x1x1x32x64xf32>
    %cst_95 = arith.constant 0.000000e+00 : f32
    %c0_96 = arith.constant 0 : index
    %c0_97 = arith.constant 0 : index
    %c36 = arith.constant 36 : index
    %76 = affine.apply #map5()
    %c1_98 = arith.constant 1 : index
    %c17_99 = arith.constant 17 : index
    %77 = affine.apply #map6()
    %c2_100 = arith.constant 2 : index
    %c64 = arith.constant 64 : index
    %78 = affine.apply #map5()
    %padded_101 = tensor.pad %collapsed_93 low[%c0_96, %c0_96, %c0_96] high[%76, %77, %78] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_95 : f32
    } : tensor<36x17x64xf32> to tensor<36x32x64xf32>
    %expanded_102 = tensor.expand_shape %padded_101 [[0], [1, 2], [3, 4]] : tensor<36x32x64xf32> into tensor<36x1x32x1x64xf32>
    %79 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_102 : tensor<36x1x32x1x64xf32>) outs(%75 : tensor<36x1x1x32x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<36x1x1x32x64xf32>
    %80 = tensor.empty() : tensor<36x1x1x32x64xf32>
    %cst_103 = arith.constant 0.000000e+00 : f32
    %c0_104 = arith.constant 0 : index
    %c0_105 = arith.constant 0 : index
    %c36_106 = arith.constant 36 : index
    %81 = affine.apply #map5()
    %c1_107 = arith.constant 1 : index
    %c64_108 = arith.constant 64 : index
    %82 = affine.apply #map5()
    %c2_109 = arith.constant 2 : index
    %c17_110 = arith.constant 17 : index
    %83 = affine.apply #map6()
    %padded_111 = tensor.pad %collapsed_94 low[%c0_104, %c0_104, %c0_104] high[%81, %82, %83] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_103 : f32
    } : tensor<36x64x17xf32> to tensor<36x64x32xf32>
    %expanded_112 = tensor.expand_shape %padded_111 [[0], [1, 2], [3, 4]] : tensor<36x64x32xf32> into tensor<36x1x64x1x32xf32>
    %84 = linalg.generic {indexing_maps = [#map7, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_112 : tensor<36x1x64x1x32xf32>) outs(%80 : tensor<36x1x1x32x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<36x1x1x32x64xf32>
    %85 = tensor.empty() : tensor<36x1x1x32x32xf32>
    %cst_113 = arith.constant 0.000000e+00 : f32
    %c0_114 = arith.constant 0 : index
    %c0_115 = arith.constant 0 : index
    %c36_116 = arith.constant 36 : index
    %86 = affine.apply #map5()
    %c1_117 = arith.constant 1 : index
    %c17_118 = arith.constant 17 : index
    %87 = affine.apply #map6()
    %c2_119 = arith.constant 2 : index
    %c17_120 = arith.constant 17 : index
    %88 = affine.apply #map6()
    %padded_121 = tensor.pad %74 low[%c0_114, %c0_114, %c0_114] high[%86, %87, %88] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_113 : f32
    } : tensor<36x17x17xf32> to tensor<36x32x32xf32>
    %expanded_122 = tensor.expand_shape %padded_121 [[0], [1, 2], [3, 4]] : tensor<36x32x32xf32> into tensor<36x1x32x1x32xf32>
    %89 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_122 : tensor<36x1x32x1x32xf32>) outs(%85 : tensor<36x1x1x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<36x1x1x32x32xf32>
    %90 = linalg.generic {indexing_maps = [#map10, #map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%79, %84 : tensor<36x1x1x32x64xf32>, tensor<36x1x1x32x64xf32>) outs(%89 : tensor<36x1x1x32x32xf32>) {
    ^bb0(%in: f32, %in_162: f32, %out: f32):
      %125 = arith.mulf %in, %in_162 : f32
      %126 = arith.addf %out, %125 : f32
      linalg.yield %126 : f32
    } -> tensor<36x1x1x32x32xf32>
    %91 = tensor.empty() : tensor<36x1x32x1x32xf32>
    %92 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%90 : tensor<36x1x1x32x32xf32>) outs(%91 : tensor<36x1x32x1x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<36x1x32x1x32xf32>
    %collapsed_123 = tensor.collapse_shape %92 [[0], [1, 2], [3, 4]] : tensor<36x1x32x1x32xf32> into tensor<36x32x32xf32>
    %extracted_slice_124 = tensor.extract_slice %collapsed_123[0, 0, 0] [36, 17, 17] [1, 1, 1] : tensor<36x32x32xf32> to tensor<36x17x17xf32>
    %expanded_125 = tensor.expand_shape %extracted_slice_124 [[0, 1], [2], [3]] : tensor<36x17x17xf32> into tensor<3x12x17x17xf32>
    %93 = tensor.empty() : tensor<3x12x17x17xf32>
    %94 = linalg.generic {indexing_maps = [#map14, #map17, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_125, %cst : tensor<3x12x17x17xf32>, tensor<f64>) outs(%93 : tensor<3x12x17x17xf32>) {
    ^bb0(%in: f32, %in_162: f64, %out: f32):
      %125 = arith.truncf %in_162 : f64 to f32
      %126 = arith.divf %in, %125 : f32
      linalg.yield %126 : f32
    } -> tensor<3x12x17x17xf32>
    %95 = tensor.empty() : tensor<3x12x17x1xf32>
    %96 = linalg.generic {indexing_maps = [#map17, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_3 : f32) outs(%95 : tensor<3x12x17x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x12x17x1xf32>
    %97 = linalg.generic {indexing_maps = [#map14, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%94 : tensor<3x12x17x17xf32>) outs(%96 : tensor<3x12x17x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %125 = arith.maxf %in, %out : f32
      linalg.yield %125 : f32
    } -> tensor<3x12x17x1xf32>
    %98 = linalg.generic {indexing_maps = [#map14, #map18, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%94, %97 : tensor<3x12x17x17xf32>, tensor<3x12x17x1xf32>) outs(%93 : tensor<3x12x17x17xf32>) {
    ^bb0(%in: f32, %in_162: f32, %out: f32):
      %125 = arith.subf %in, %in_162 : f32
      linalg.yield %125 : f32
    } -> tensor<3x12x17x17xf32>
    %99 = linalg.generic {indexing_maps = [#map14, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%98 : tensor<3x12x17x17xf32>) outs(%93 : tensor<3x12x17x17xf32>) {
    ^bb0(%in: f32, %out: f32):
      %125 = math.exp %in : f32
      linalg.yield %125 : f32
    } -> tensor<3x12x17x17xf32>
    %100 = linalg.generic {indexing_maps = [#map17, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%cst_2 : f32) outs(%95 : tensor<3x12x17x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x12x17x1xf32>
    %101 = linalg.generic {indexing_maps = [#map14, #map18], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%99 : tensor<3x12x17x17xf32>) outs(%100 : tensor<3x12x17x1xf32>) {
    ^bb0(%in: f32, %out: f32):
      %125 = arith.addf %in, %out : f32
      linalg.yield %125 : f32
    } -> tensor<3x12x17x1xf32>
    %102 = linalg.generic {indexing_maps = [#map14, #map18, #map14], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%99, %101 : tensor<3x12x17x17xf32>, tensor<3x12x17x1xf32>) outs(%93 : tensor<3x12x17x17xf32>) {
    ^bb0(%in: f32, %in_162: f32, %out: f32):
      %125 = arith.divf %in, %in_162 : f32
      linalg.yield %125 : f32
    } -> tensor<3x12x17x17xf32>
    %collapsed_126 = tensor.collapse_shape %102 [[0, 1], [2], [3]] : tensor<3x12x17x17xf32> into tensor<36x17x17xf32>
    %collapsed_127 = tensor.collapse_shape %69 [[0, 1], [2], [3]] : tensor<3x12x17x64xf32> into tensor<36x17x64xf32>
    %103 = tensor.empty() : tensor<36x17x64xf32>
    %104 = linalg.generic {indexing_maps = [#map4, #map3], iterator_types = ["parallel", "parallel", "parallel"]} ins(%cst_2 : f32) outs(%103 : tensor<36x17x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<36x17x64xf32>
    %105 = tensor.empty() : tensor<36x1x1x32x32xf32>
    %cst_128 = arith.constant 0.000000e+00 : f32
    %c0_129 = arith.constant 0 : index
    %c0_130 = arith.constant 0 : index
    %c36_131 = arith.constant 36 : index
    %106 = affine.apply #map5()
    %c1_132 = arith.constant 1 : index
    %c17_133 = arith.constant 17 : index
    %107 = affine.apply #map6()
    %c2_134 = arith.constant 2 : index
    %c17_135 = arith.constant 17 : index
    %108 = affine.apply #map6()
    %padded_136 = tensor.pad %collapsed_126 low[%c0_129, %c0_129, %c0_129] high[%106, %107, %108] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_128 : f32
    } : tensor<36x17x17xf32> to tensor<36x32x32xf32>
    %expanded_137 = tensor.expand_shape %padded_136 [[0], [1, 2], [3, 4]] : tensor<36x32x32xf32> into tensor<36x1x32x1x32xf32>
    %109 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_137 : tensor<36x1x32x1x32xf32>) outs(%105 : tensor<36x1x1x32x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<36x1x1x32x32xf32>
    %110 = tensor.empty() : tensor<36x1x1x64x32xf32>
    %cst_138 = arith.constant 0.000000e+00 : f32
    %c0_139 = arith.constant 0 : index
    %c0_140 = arith.constant 0 : index
    %c36_141 = arith.constant 36 : index
    %111 = affine.apply #map5()
    %c1_142 = arith.constant 1 : index
    %c17_143 = arith.constant 17 : index
    %112 = affine.apply #map6()
    %c2_144 = arith.constant 2 : index
    %c64_145 = arith.constant 64 : index
    %113 = affine.apply #map5()
    %padded_146 = tensor.pad %collapsed_127 low[%c0_139, %c0_139, %c0_139] high[%111, %112, %113] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_138 : f32
    } : tensor<36x17x64xf32> to tensor<36x32x64xf32>
    %expanded_147 = tensor.expand_shape %padded_146 [[0], [1, 2], [3, 4]] : tensor<36x32x64xf32> into tensor<36x1x32x1x64xf32>
    %114 = linalg.generic {indexing_maps = [#map7, #map9], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_147 : tensor<36x1x32x1x64xf32>) outs(%110 : tensor<36x1x1x64x32xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<36x1x1x64x32xf32>
    %115 = tensor.empty() : tensor<36x1x1x32x64xf32>
    %cst_148 = arith.constant 0.000000e+00 : f32
    %c0_149 = arith.constant 0 : index
    %c0_150 = arith.constant 0 : index
    %c36_151 = arith.constant 36 : index
    %116 = affine.apply #map5()
    %c1_152 = arith.constant 1 : index
    %c17_153 = arith.constant 17 : index
    %117 = affine.apply #map6()
    %c2_154 = arith.constant 2 : index
    %c64_155 = arith.constant 64 : index
    %118 = affine.apply #map5()
    %padded_156 = tensor.pad %104 low[%c0_149, %c0_149, %c0_149] high[%116, %117, %118] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index):
      tensor.yield %cst_148 : f32
    } : tensor<36x17x64xf32> to tensor<36x32x64xf32>
    %expanded_157 = tensor.expand_shape %padded_156 [[0], [1, 2], [3, 4]] : tensor<36x32x64xf32> into tensor<36x1x32x1x64xf32>
    %119 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%expanded_157 : tensor<36x1x32x1x64xf32>) outs(%115 : tensor<36x1x1x32x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<36x1x1x32x64xf32>
    %120 = linalg.generic {indexing_maps = [#map10, #map11, #map12], iterator_types = ["parallel", "parallel", "parallel", "reduction", "parallel", "parallel", "reduction"]} ins(%109, %114 : tensor<36x1x1x32x32xf32>, tensor<36x1x1x64x32xf32>) outs(%119 : tensor<36x1x1x32x64xf32>) {
    ^bb0(%in: f32, %in_162: f32, %out: f32):
      %125 = arith.mulf %in, %in_162 : f32
      %126 = arith.addf %out, %125 : f32
      linalg.yield %126 : f32
    } -> tensor<36x1x1x32x64xf32>
    %121 = tensor.empty() : tensor<36x1x32x1x64xf32>
    %122 = linalg.generic {indexing_maps = [#map7, #map8], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%120 : tensor<36x1x1x32x64xf32>) outs(%121 : tensor<36x1x32x1x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<36x1x32x1x64xf32>
    %collapsed_158 = tensor.collapse_shape %122 [[0], [1, 2], [3, 4]] : tensor<36x1x32x1x64xf32> into tensor<36x32x64xf32>
    %extracted_slice_159 = tensor.extract_slice %collapsed_158[0, 0, 0] [36, 17, 64] [1, 1, 1] : tensor<36x32x64xf32> to tensor<36x17x64xf32>
    %expanded_160 = tensor.expand_shape %extracted_slice_159 [[0, 1], [2], [3]] : tensor<36x17x64xf32> into tensor<3x12x17x64xf32>
    %123 = tensor.empty() : tensor<3x17x12x64xf32>
    %124 = linalg.generic {indexing_maps = [#map14, #map15], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%expanded_160 : tensor<3x12x17x64xf32>) outs(%123 : tensor<3x17x12x64xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
    } -> tensor<3x17x12x64xf32>
    %collapsed_161 = tensor.collapse_shape %124 [[0], [1], [2, 3]] : tensor<3x17x12x64xf32> into tensor<3x17x768xf32>
    return %collapsed_161 : tensor<3x17x768xf32>
  }
}

