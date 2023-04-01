//===- CUDNNTypes.h - CUDNN dialect types ------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CUDNN_CUDNNTYPES_H
#define CUDNN_CUDNNTYPES_H

#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "openxla/compiler/nvgpu/Dialect/CUDNN/IR/CUDNNTypes.h.inc"

#endif // CUDNN_CUDNNTYPES_H
