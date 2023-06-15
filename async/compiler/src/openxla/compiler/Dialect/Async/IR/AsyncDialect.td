// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef ASYNC_DIALECT
#define ASYNC_DIALECT

include "mlir/IR/OpBase.td"

def AsyncDialect : Dialect {
  let name = "async";
  let cppNamespace = "::openxla::compiler::async";

  let summary = "Types and operations for async dialect";
  let description = [{
    This dialect contains operations for modeling asynchronous execution.
  }];

  let useDefaultTypePrinterParser = 1;
}

#endif // ASYNC_DIALECT
