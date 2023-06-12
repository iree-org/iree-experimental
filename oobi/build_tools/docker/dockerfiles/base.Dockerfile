# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image that includes Tensorflow 2.12.0 with GPU support.

# Ubuntu 22.04.
FROM ubuntu@sha256:817cfe4672284dcbfee885b1a66094fd907630d610cab329114d036716be49ba

######## Base ########
RUN apt-get update \
  && apt-get install -y \
    cmake \
    ninja-build \
    git \
    unzip \
    wget \
    curl \
    gnupg2 \
    python3-numpy

######## Clang/LLVM ########
ARG LLVM_VERSION=14

RUN apt-get update \
  && apt-get install -y \
    llvm-${LLVM_VERSION} \
    llvm-${LLVM_VERSION}-dev \
    clang-${LLVM_VERSION} \
    clang-tools-${LLVM_VERSION} \
    libclang-common-${LLVM_VERSION}-dev \
    libclang-${LLVM_VERSION}-dev \
    libclang1-${LLVM_VERSION} \
    clang-format-${LLVM_VERSION} \
    clangd-${LLVM_VERSION} \
    clang-tidy-${LLVM_VERSION} \
    lldb-${LLVM_VERSION} \
    lld-${LLVM_VERSION} \
    libmlir-${LLVM_VERSION}-dev \
    mlir-${LLVM_VERSION}-tools \
  && ln -s /usr/lib/llvm-${LLVM_VERSION}/bin/clang /usr/bin/clang \
  && ln -s /usr/lib/llvm-${LLVM_VERSION}/bin/clang++ /usr/bin/clang++ \
  && ln -s /usr/bin/lld-${LLVM_VERSION} /usr/bin/lld

ENV CC /usr/bin/clang
ENV CXX /usr/bin/clang++
##############

######## Python ########
WORKDIR /install-python

ARG PYTHON_VERSION=3.10

COPY python_build_requirements.txt install_python_deps.sh ./
RUN ./install_python_deps.sh "${PYTHON_VERSION}" \
  && apt-get -y install python-is-python3 \
  && rm -rf /install-python

ENV PYTHON_BIN /usr/bin/python3

WORKDIR /

######## Bazel ########
WORKDIR /install-bazel
COPY install_bazel.sh .bazelversion ./
RUN ./install_bazel.sh && rm -rf /install-bazel
WORKDIR /
