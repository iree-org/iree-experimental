// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

CUDNN_PFN_DECL(cudnnCreate, cudnnHandle_t*)
CUDNN_PFN_DECL(cudnnDestroy, cudnnHandle_t)
CUDNN_PFN_DECL_STR_RETURN(cudnnGetErrorString)
