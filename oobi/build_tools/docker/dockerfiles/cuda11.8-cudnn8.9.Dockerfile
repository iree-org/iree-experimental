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
    git \
    unzip \
    wget \
    curl \
    gnupg2 \
    python3-numpy

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

######## NVIDIA ########
WORKDIR /install-cuda

# Install CUDA Toolkit. Instructions from https://developer.nvidia.com/cuda-downloads.
RUN wget "https://storage.googleapis.com/iree-shared-files/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb" \
  && dpkg --install "cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb" \
  && cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/ \
  && apt-get update \
  && apt-get -y install cuda-toolkit-11-8

ENV PATH="/usr/local/cuda-11.8/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-11.8/lib64:${LD_LIBRARY_PATH}"

# Install CuDNN. Instructions from https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html.
RUN wget "https://storage.googleapis.com/iree-shared-files/cudnn-local-repo-ubuntu2204-8.9.0.131_1.0-1_amd64.deb" \
  && dpkg --install "cudnn-local-repo-ubuntu2204-8.9.0.131_1.0-1_amd64.deb" \
  && cp /var/cudnn-local-repo-ubuntu2204-8.9.0.131/cudnn-*-keyring.gpg /usr/share/keyrings/ \
  && apt-get update \
  && apt-get -y install libcudnn8 \
  && apt-get -y install libcudnn8-dev \
  && rm -rf /install-cuda

WORKDIR /
