// RUN: iree-compile %s --iree-hal-target-backends=cuda | openxla-runner - example.main | FileCheck %s

module @example {

  func.func private @cudnn.tensor.arg(
    %dtype: i64, %dims: !util.list<i64>, %uid: i64, %alignment: i64
  ) -> !cudnn.tensor

  func.func private @cudnn.pointwise_relu(
    %input: !cudnn.tensor, %lower: f32, %upper: f32, %uid: i64, %alignment: i64
  ) -> !cudnn.tensor

  func.func private @cudnn.tensor.debug(%arg0: !cudnn.tensor)

  func.func @main() {
    %rank = arith.constant 4 : index
    
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c3 = arith.constant 3 : index
    
    %c128 = arith.constant 128 : i64

    // Tensor UIDs
    %uid0 = arith.constant 0 : i64
    %uid1 = arith.constant 1 : i64
    
    // [128, 128, 128, 128]
    %dims = util.list.create %rank : !util.list<i64>
    util.list.resize %dims, %rank : !util.list<i64>
    util.list.set %dims[%c0], %c128 : !util.list<i64>
    util.list.set %dims[%c1], %c128 : !util.list<i64>
    util.list.set %dims[%c2], %c128 : !util.list<i64>
    util.list.set %dims[%c3], %c128 : !util.list<i64>

    // CUDNN_DATA_FLOAT
    %dtype = arith.constant 0 : i64

    // Tensor alignment
    %alignment = arith.constant 32 : i64

    // Create !cudnn.tensor<128x128x128x128xf32>
    %0 = call @cudnn.tensor.arg(%dtype, %dims, %uid0, %alignment)
           : (i64, !util.list<i64>, i64, i64) -> !cudnn.tensor

    // Create pointwise relu operation
    %lower = arith.constant 0.0 : f32
    %upper = arith.constant 9.0 : f32
    %1 = call @cudnn.pointwise_relu(%0, %lower, %upper, %uid1, %alignment)
           : (!cudnn.tensor, f32, f32, i64, i64) -> !cudnn.tensor

    // CHECK: CUDNN_BACKEND_TENSOR_DESCRIPTOR : Datatype: CUDNN_DATA_FLOAT
    // CHECK: Id: 123
    // CHECK: Alignment: 32
    // CHECK: nDims 4
    // CHECK: VectorCount: 1
    // CHECK: vectorDimension -1
    // CHECK: Dim [ 128,128,128,128 ]
    // CHECK: Str [ 2097152,16384,128,1 ]
    // CHECK: isVirtual: 0
    // CHECK: isByValue: 0
    // CHECK: reorder_type: CUDNN_TENSOR_REORDERING_NONE
    call @cudnn.tensor.debug(%0) : (!cudnn.tensor) -> ()
    call @cudnn.tensor.debug(%1) : (!cudnn.tensor) -> ()

    return
  }

}
