# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

iree_cc_binary(
  NAME
    openxla-runner
  SRCS
    "openxla-runner.c"
  DEPS
    iree::base
    iree::base::internal::flags
    iree::base::tracing
    iree::hal
    iree::modules::hal::types
    iree::tooling::comparison
    iree::tooling::context_util
    iree::tooling::device_util
    iree::tooling::instrument_util
    iree::tooling::vm_util
    iree::vm
    openxla::runtime::nvgpu::cudnn_module
)
