#!/bin/bash

DEVICE=$1
OUTPUT_DIR=$2
CUDA_VERSION=${3:-"11.8"}

git clone https://github.com/openxla/xla.git
pushd xla

if [ "${DEVICE}" = "gpu" ]; then
  bazel build -c opt --config=cuda \
    --action_env TF_CUDA_COMPUTE_CAPABILITIES="8.0" \
    --action_env GCC_HOST_COMPILER_PATH="/usr/bin/x86_64-linux-gnu-gcc-11" \
    --action_env LD_LIBRARY_PATH="/usr/local/cuda-${CUDA_VERSION}/lib64:" \
    --action_env CUDA_TOOLKIT_PATH="/usr/local/cuda-${CUDA_VERSION}" \
    --copt=-Wno-switch \
    xla/tools/multihost_hlo_runner:hlo_runner_main
  RUN_HLO_MODULE_PATH=$(realpath "bazel-bin/xla/tools/multihost_hlo_runner/hlo_runner_main")
else
  bazel build -c opt --copt=-Wno-switch \
    --action_env GCC_HOST_COMPILER_PATH="/usr/bin/x86_64-linux-gnu-gcc-11" \
    xla/tools:run_hlo_module
  RUN_HLO_MODULE_PATH=$(realpath "bazel-bin/xla/tools/run_hlo_module")
fi

popd

cp "${RUN_HLO_MODULE_PATH}" "${OUTPUT_DIR}/"
