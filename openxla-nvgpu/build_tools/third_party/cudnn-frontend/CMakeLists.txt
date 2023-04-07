# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

set(CUDNN_FRONTEND_ROOT "${OPENXLA_NVGPU_ROOT_DIR}/third_party/cudnn-frontend/")

external_cc_library(
  PACKAGE
    cudnn_frontend
  NAME
    cudnn_frontend
  ROOT
    ${CUDNN_FRONTEND_ROOT}
  INCLUDES
    "${CUDNN_FRONTEND_ROOT}/include"    
  HDRS
    "include/cudnn_frontend.h"
  PUBLIC
)
