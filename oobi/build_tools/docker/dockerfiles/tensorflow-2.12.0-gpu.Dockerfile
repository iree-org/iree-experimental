# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image that includes Tensorflow 2.12.0 with GPU support.

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04@sha256:f309b8d1bfc75b291ab53cb8677a681b3eb6664ee52f4e21e4eb17da0a48c3d4

######## Python ########
WORKDIR /install-Python

ARG PYTHON_VERSION=3.10

COPY python_build_requirements.txt install_python_deps.sh ./
RUN ./install_python_deps.sh "${PYTHON_VERSION}" \
  && rm -rf /install-python

ENV PYTHON_BIN /usr/bin/python3
##############

######## Bazel ########
WORKDIR /install-bazel
COPY install_bazel.sh .bazelversion ./
RUN ./install_bazel.sh && rm -rf /install-bazel
##############
