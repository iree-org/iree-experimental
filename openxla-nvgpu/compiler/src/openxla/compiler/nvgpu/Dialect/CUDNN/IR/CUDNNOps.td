//===- CUDNNOps.td - CUDNN dialect ops -----------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDNN_OPS
#define CUDNN_OPS

include "mlir/IR/OpBase.td"
include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.td"
include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

// Pointwise operations
// --------------------
def CUDNN_PointWiseAddOp : CUDNN_Op<"pointwise_add", [Pure]> {
    let summary = "Pointwise Add";

    let arguments = (ins
      CUDNN_TensorDescType:$input,
      TypeAttr:$compute_type
    );
    let results = (outs CUDNN_TensorDescType:$res);

    let assemblyFormat = [{
      `(` $input `)` `type` `=` $compute_type
        attr-dict `:` qualified(type($input)) `->` qualified(type($res))
    }];
}

def CUDNN_PointWiseReluOp : CUDNN_Op<"pointwise_relu", [Pure]> {
    let summary = "Pointwise Relu";

    let arguments = (ins
      CUDNN_TensorDescType:$input,
      TypeAttr:$compute_type,
      F64Attr:$lower_clip
    );
    let results = (outs CUDNN_TensorDescType:$res);

    let assemblyFormat = [{
      `(` $input `)` `type` `=` $compute_type `lower_clip` `=` $lower_clip
        attr-dict `:` qualified(type($input)) `->` qualified(type($res))
    }];
}

// Convolutions
// ------------
def CUDNN_ConvolutionOp : CUDNN_Op<"convolution", [Pure]> {
    let summary = "Convolution";

    let arguments = (ins
      CUDNN_TensorDescType:$x,
      CUDNN_TensorDescType:$w,
      TypeAttr:$element_type,
      F32Attr:$alpha,
      F32Attr:$beta,
      I32Attr:$spatial_dim_count,
      DenseI64ArrayAttr:$spatial_stride,
      DenseI64ArrayAttr:$pre_padding,
      DenseI64ArrayAttr:$post_padding,
      DenseI64ArrayAttr:$dilation
    );
    let results = (outs CUDNN_TensorDescType:$y);

    let assemblyFormat = [{
      `(` $x `,` $w `)` `type` `=` $element_type
         `alpha` `=` $alpha
         `beta` `=` $beta
         `spatial_dim_count` `=` $spatial_dim_count
         `spatial_stride` `=`  $spatial_stride
         `pre_padding` `=` $pre_padding
         `post_padding` `=` $post_padding
         `dilation` `=` $dilation
         attr-dict `:` qualified(type($x)) `,` qualified(type($w)) `->` qualified(type($y))
    }];
}

def CUDNN_CrossCorrelationOp : CUDNN_Op<"cross_correlation", [Pure]> {
    let summary = "Cross correlation";

    let arguments = (ins
      CUDNN_TensorDescType:$x,
      CUDNN_TensorDescType:$w,
      TypeAttr:$element_type,
      F32Attr:$alpha,
      F32Attr:$beta,
      I32Attr:$spatial_dim_count,
      DenseI64ArrayAttr:$spatial_stride,
      DenseI64ArrayAttr:$pre_padding,
      DenseI64ArrayAttr:$post_padding,
      DenseI64ArrayAttr:$dilation
    );
    let results = (outs CUDNN_TensorDescType:$y);

    let assemblyFormat = [{
      `(` $x `,` $w `)` `type` `=` $element_type
          `alpha` `=` $alpha
          `beta` `=` $beta
          `spatial_dim_count` `=` $spatial_dim_count
          `spatial_stride` `=`  $spatial_stride
          `pre_padding` `=` $pre_padding
          `post_padding` `=` $post_padding
          `dilation` `=` $dilation
          attr-dict `:` qualified(type($x)) `,` qualified(type($w)) `->` qualified(type($y))
    }];
}

// MatMul
// ------
def CUDNN_MatMulOp : CUDNN_Op<"matmul", [Pure]> {
    let summary = "Matmul";

    let arguments = (ins
      CUDNN_TensorDescType:$a,
      CUDNN_TensorDescType:$b,
      TypeAttr:$element_type
    );
    let results = (outs CUDNN_TensorDescType:$c);

    let assemblyFormat = [{
      `(` $a `,` $b `)` `type` `=` $element_type
         attr-dict `:` qualified(type($a)) `,` qualified(type($b)) `->` qualified(type($c))
    }];
}

// Reduction
// ---------
def CUDNN_ReductionOp : CUDNN_Op<"reduction", [Pure]> {
    let summary = "Reduction";

    let arguments = (ins
      CUDNN_TensorDescType:$x,
      // TODO: Shouldn't be enum.
      I32Attr:$reduction_op,
      TypeAttr:$element_type
    );
    let results = (outs CUDNN_TensorDescType:$y);

    let assemblyFormat = [{
       `(` $x `)` `type` `=` $element_type `reduction_op` `=` $reduction_op
         attr-dict `:` qualified(type($x)) `->` qualified(type($y))
    }];
}

// Utility functions.
// ------------------
def CUDNN_BuildGraphOp : CUDNN_Op<"build_graph", [Pure, Terminator] > {
    let summary = "Build compute graph descriptor";

    let arguments = (ins
      Variadic<CUDNN_TensorDescType>:$graph_results
    );

    let assemblyFormat = [{
        `(` $graph_results `)` attr-dict `:` type($graph_results)
    }];
}

def CUDNN_BuildAndExecGraphOp : CUDNN_Op<"build_and_exec_graph", [Pure, IsolatedFromAbove]> {
    let summary = "Build and exec graph";

    let arguments = (ins
      CUDNN_HandleType:$handle,
      Variadic<RankedTensorOf<[AnyFloat]>>:$inputs
    );

    let results = (outs
      Variadic<RankedTensorOf<[AnyFloat]>>:$outputs
    );

    let regions = (region
      SizedRegion<1>:$constructor
    );

     let assemblyFormat = [{
      $handle `(` $inputs `)` attr-dict-with-keyword $constructor `:`
        type($inputs) `->` type($outputs)
     }];
 }

def CUDNN_GetCurrentHandle : CUDNN_Op<"get_current_handle", [Pure]> {
    let summary = "Get current handle";

    let arguments = (ins);
    let results = (outs CUDNN_HandleType:$handle);
    let assemblyFormat = [{ attr-dict type($handle) }];
}

#endif // CUDNN_OPS
