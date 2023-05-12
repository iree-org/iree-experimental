#!/bin/bash

COMMIT_SHA=$1
CUDA_VERSION=$2

echo "LD_LIBRARY_PATH: $(echo ${LD_LIBRARY_PATH})"

git clone https://github.com/tensorflow/tensorflow.git
pushd tensorflow
git checkout "${COMMIT_SHA}"

if [[ -n "${CUDA_VERSION}" ]]; then
  bazel build -c opt --config=cuda \
    --action_env TF_CUDA_COMPUTE_CAPABILITIES="8.0" \
    --action_env GCC_HOST_COMPILER_PATH="/usr/bin/x86_64-linux-gnu-gcc-11" \
    --action_env LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:" \
    --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda-${CUDA_VERSION}" \
    --copt=-Wno-switch \
    tensorflow/compiler/xla/tools/run_hlo_module
else
  bazel build -c opt --copt=-Wno-switch \
    --action_env GCC_HOST_COMPILER_PATH="/usr/bin/x86_64-linux-gnu-gcc-11" \
    tensorflow/compiler/xla/tools/run_hlo_module
fi

popd
