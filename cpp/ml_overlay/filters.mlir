func.func @filter_add(
  %source: tensor<?x?x4xi8>
) -> (tensor<?x?x4xi8>) {
  %0 = arith.addi %source, %source : tensor<?x?x4xi8>
  return %0 : tensor<?x?x4xi8>
}

func.func @filter_add_inplace(
  %source: tensor<?x?x4xi8>,
  %target_storage: !hal.buffer {iree.abi.output = 0 : index}
) -> (tensor<?x?x4xi8>) {
  %0 = arith.addi %source, %source : tensor<?x?x4xi8>
  return %0 : tensor<?x?x4xi8>
}

func.func @filter_cst(
  %source: tensor<?x?x4xi8>
) -> (tensor<?x?x4xi8>) {
  %0 = arith.constant dense<[[[255, 128, 0, 255]]]> : tensor<1x1x4xi8>
  %1 = shape.shape_of %source : tensor<?x?x4xi8> -> tensor<3xindex>
  %2 = "mhlo.dynamic_broadcast_in_dim"(%0, %1) {broadcast_dimensions = dense<[0, 1, 2]> : tensor<3xi64>} : (tensor<1x1x4xi8>, tensor<3xindex>) -> tensor<?x?x4xi8>
  return %2 : tensor<?x?x4xi8>
}

func.func @filter_bboxes(
  %source: tensor<?x?x4xi8>
) -> (tensor<?x?x4xi8>, tensor<?x16xi32>) {
  %0 = arith.addi %source, %source : tensor<?x?x4xi8>
  // tensor<?x16xi32>
  // id, data0, data1, data2 | x0, y0, x1, y1 | a, b, g, r | x x x x
  %1 = arith.constant dense<[
    [1000, 100, 101, 102,
     20, 40, 50, 60,
     255, 255, 0, 0,
     0, 0, 0, 0],
    [2000, 200, 201, 202,
     200, 240, 350, 460,
     255, 0, 0, 255,
     0, 0, 0, 0],
    [3000, 300, 301, 302,
     100, 340, 250, 360,
     128, 255, 255, 255,
     0, 0, 0, 0]
  ]> : tensor<3x16xi32>
  %2 = tensor.cast %1 : tensor<3x16xi32> to tensor<?x16xi32>
  return %0, %2 : tensor<?x?x4xi8>, tensor<?x16xi32>
}
