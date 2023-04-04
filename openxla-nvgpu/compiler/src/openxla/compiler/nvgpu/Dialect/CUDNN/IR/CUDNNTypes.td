//===- CUDNNTypes.td - CUDNN dialect types ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDNN_TYPES
#define CUDNN_TYPES

include "mlir/IR/AttrTypeBase.td"
include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNDialect.td"

//===----------------------------------------------------------------------===//
// CUDNN type definitions
//===----------------------------------------------------------------------===//

class CUDNN_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<CUDNN_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def CUDNN_ExecutionPlanType : CUDNN_Type<"ExecutionPlan", "execution_plan"> {
    let summary = "CUDNN Execution Plan";
}

def CUDNN_OperationGraphType : CUDNN_Type<"OperationGraph", "operation_graph"> {
    let summary = "CUDNN Operation Graph";
    let description = [{
      Handle to graph of operations that will be performed.
    }];
}

def CUDNN_HandleType : CUDNN_Type<"Handle", "handle"> {
    let summary = "CUDNN handle";
}

def CUDNN_StatusType : CUDNN_Type<"Status", "status"> {
    let summary = "CUDNN Status type";
    let description = "Return code, corresponding to cudnnStatus_t";
}

def CUDNN_TensorDescType : CUDNN_Type<"TensorDesc", "tensor_desc"> {
    let summary = "CUDNN Tensor descriptor";
    let description = [{
      Description of the memory layout, element type, alignment and stride of
      the input/results.
    }];
    let parameters = (ins
      ArrayRefParameter<"int64_t">:$shape,
      "Type":$element_type,
      "int":$alignment,
       ArrayRefParameter<"int64_t">:$stride
    );
    // TODO: This is currently order dependent, remove the need for named parts.
    let assemblyFormat = [{
      `<` custom<DimensionList>($shape, $element_type) `,` `alignment` `=` $alignment `,`
      `stride` `=` `[` $stride `]` `>`
    }];
}

#endif // CUDNN_TYPES
