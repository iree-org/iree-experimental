// Copyright 2023 The OpenXLA Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef OPENXLA_COMPILER_DIALECT_ASYNC_CONVERSION_CONVERT_ASYNC_TO_RUNTIME_H_
#define OPENXLA_COMPILER_DIALECT_ASYNC_CONVERSION_CONVERT_ASYNC_TO_RUNTIME_H_

#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

namespace openxla::compiler::async {

// Appends Async dialect to async runtime patterns to the given pattern list.
// Conversion patterns lower from Async dialect operations to function calls
// corresponding to the async runtime (implemented as a custom VM module).
void populateAsyncToRuntimePatterns(mlir::TypeConverter &typeConverter,
                                    mlir::RewritePatternSet &patterns);

}  // namespace openxla::compiler::async

#endif  // OPENXLA_COMPILER_DIALECT_ASYNC_CONVERSION_CONVERT_ASYNC_TO_RUNTIME_H_
