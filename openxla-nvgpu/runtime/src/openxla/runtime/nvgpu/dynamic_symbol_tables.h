// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
// Functions required for setting up cuDNN in CuDNNModule.
//===----------------------------------------------------------------------===//

CUDNN_PFN_DECL(cudnnCreate, cudnnHandle_t *)
CUDNN_PFN_DECL(cudnnDestroy, cudnnHandle_t)
CUDNN_PFN_DECL_STR_RETURN(cudnnGetErrorString)

//===----------------------------------------------------------------------===//
// Functions required for compiling cudnn_frontend (see cudnn_tensor.{h,cpp}).
//===----------------------------------------------------------------------===//

CUDNN_PFN_DECL(cudnnBackendFinalize, cudnnBackendDescriptor_t)
CUDNN_PFN_DECL(cudnnBackendSetAttribute, cudnnBackendDescriptor_t,
               cudnnBackendAttributeName_t, cudnnBackendAttributeType_t,
               int64_t, const void *)
CUDNN_PFN_DECL(cudnnBackendCreateDescriptor, cudnnBackendDescriptorType_t,
               cudnnBackendDescriptor_t *)
CUDNN_PFN_DECL(cudnnBackendDestroyDescriptor, cudnnBackendDescriptor_t)