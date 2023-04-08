// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "openxla/runtime/nvgpu/cudnn_stub.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

//===----------------------------------------------------------------------===//
// cuDNN API required for compiling cudnn_frontend.
//===----------------------------------------------------------------------===//

cudnnStatus_t CUDNNWINAPI
cudnnBackendSetAttribute(cudnnBackendDescriptor_t descriptor,
                         cudnnBackendAttributeName_t attributeName,
                         cudnnBackendAttributeType_t attributeType,
                         int64_t elementCount, const void* arrayOfElements) {
  auto* syms = openxla::runtime::nvgpu::ScopedCuDNNStubs::syms();
  IREE_ASSERT(syms);
  return syms->cudnnBackendSetAttribute(
      descriptor, attributeName, attributeType, elementCount, arrayOfElements);
}

cudnnStatus_t CUDNNWINAPI
cudnnBackendCreateDescriptor(cudnnBackendDescriptorType_t descriptorType,
                             cudnnBackendDescriptor_t* descriptor) {
  auto* syms = openxla::runtime::nvgpu::ScopedCuDNNStubs::syms();
  IREE_ASSERT(syms);
  return syms->cudnnBackendCreateDescriptor(descriptorType, descriptor);
}

cudnnStatus_t CUDNNWINAPI
cudnnBackendDestroyDescriptor(cudnnBackendDescriptor_t descriptor) {
  auto* syms = openxla::runtime::nvgpu::ScopedCuDNNStubs::syms();
  IREE_ASSERT(syms);
  return syms->cudnnBackendDestroyDescriptor(descriptor);
}

cudnnStatus_t CUDNNWINAPI
cudnnBackendFinalize(cudnnBackendDescriptor_t descriptor) {
  auto* syms = openxla::runtime::nvgpu::ScopedCuDNNStubs::syms();
  IREE_ASSERT(syms);
  return syms->cudnnBackendFinalize(descriptor);
}

size_t CUDNNWINAPI cudnnGetVersion() {
  auto* syms = openxla::runtime::nvgpu::ScopedCuDNNStubs::syms();
  IREE_ASSERT(syms);
  return syms->cudnnGetVersion();
}

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus
