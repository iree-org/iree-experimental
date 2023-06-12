#!/bin/bash

DEVICE=$1
OUTPUT_PATH=$2
CUDA_SDK_DIR=${CUDA_SDK_DIR:-/usr/local/cuda}

TD="$(cd $(dirname $0) && pwd)"

function die() {
  echo "Error executing command: $*"
  exit 1
}

# Create a subdirectory for all repos to be clone into.
rm -rf github
mkdir github
cd github

# Create virtual environment.
VENV_DIR=${VENV_DIR:-pjrt.venv}
python3 -m venv "${VENV_DIR}" || die "Could not create venv."
source "${VENV_DIR}/bin/activate" || die "Could not activate venv"
python3 -m pip install --upgrade pip || die "Could not upgrade pip"

# Build pjrt plugin.
git clone https://github.com/openxla/openxla-pjrt-plugin.git
pushd openxla-pjrt-plugin

python3 ./sync_deps.py
python3 -m pip install -U -r requirements.txt
python3 -m pip install -U requests

if [[ ! -d "${CUDA_SDK_DIR}" ]]; then
  CUDA_SDK_DIR=${HOME?}/.iree_cuda_deps
  ../iree/build_tools/docker/context/fetch_cuda_deps.sh ${CUDA_SDK_DIR?}
fi

python3 ./configure.py --cc=clang --cxx=clang++ --cuda-sdk-dir=$CUDA_SDK_DIR

source .env.sh

echo "IREE_PJRT_COMPILER_LIB_PATH: ${IREE_PJRT_COMPILER_LIB_PATH}"
echo "PJRT_NAMES_AND_LIBRARY_PATHS: ${PJRT_NAMES_AND_LIBRARY_PATHS}"
echo "IREE_CUDA_DEPS_DIR: ${IREE_CUDA_DEPS_DIR}"

# Build.
bazel build iree/integrations/pjrt/...

popd

MODEL_RESNET50_FP32_JAX="aff75509-4420-40a8-844e-dbfc48494fe6-MODEL_RESNET50-fp32-JAX-3x224x224xf32"
MODEL_BERT_LARGE_FP32_JAX="47cb0d3a-5eb7-41c7-9d7c-97aae7023ecf-MODEL_BERT_LARGE-fp32-JAX-384xi32"
MODEL_T5_LARGE_FP32_JAX="173c7180-bad4-4b91-8423-4beeb13d2b0a-MODEL_T5_LARGE-fp32-JAX-512xi32"

declare -a gpu_benchmark_ids=(
  #"${MODEL_RESNET50_FP32_JAX}-batch1"
  #"${MODEL_RESNET50_FP32_JAX}-batch8"
  #"${MODEL_RESNET50_FP32_JAX}-batch64"
  #"${MODEL_RESNET50_FP32_JAX}-batch128"
  #"${MODEL_RESNET50_FP32_JAX}-batch256"
  #"${MODEL_RESNET50_FP32_JAX}-batch2048"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch1"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch16"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch24"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch32"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch48"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch64"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch512"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch1024"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch1280"
  "${MODEL_T5_LARGE_FP32_JAX}-batch1"
  "${MODEL_T5_LARGE_FP32_JAX}-batch16"
  "${MODEL_T5_LARGE_FP32_JAX}-batch24"
  "${MODEL_T5_LARGE_FP32_JAX}-batch32"
  "${MODEL_T5_LARGE_FP32_JAX}-batch48"
  "${MODEL_T5_LARGE_FP32_JAX}-batch64"
  "${MODEL_T5_LARGE_FP32_JAX}-batch512"
)

declare -a cpu_benchmark_ids=(
  "${MODEL_RESNET50_FP32_JAX}-batch1"
  "${MODEL_RESNET50_FP32_JAX}-batch64"
  "${MODEL_RESNET50_FP32_JAX}-batch128"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch1"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch32"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch64"
  "${MODEL_T5_LARGE_FP32_JAX}-batch1"
  "${MODEL_T5_LARGE_FP32_JAX}-batch16"
  "${MODEL_T5_LARGE_FP32_JAX}-batch32"
)

if [ "${DEVICE}" = "gpu" ]; then
    BENCHMARK_IDS=("${gpu_benchmark_ids[@]}")
    ITERATIONS=50
    DEVICE="iree_cuda"
else
    BENCHMARK_IDS=("${cpu_benchmark_ids[@]}")
    ITERATIONS=20
    DEVICE="iree_cpu"
fi

# Create json file and populate with global information.
rm "${OUTPUT_PATH}"
echo "{\"trigger\": { \"timestamp\": \"$(date +'%s')\" }, \"benchmarks\": []}" > "${OUTPUT_PATH}"

CACHE_DIR="${OOBI_CACHE_DIR:-/tmp/oobi/cache}"
mkdir -p "${CACHE_DIR}"

python3 -m pip install --upgrade flax
python3 -m pip install --upgrade transformers
python3 -m pip install --upgrade pillow

for benchmark_id in "${BENCHMARK_IDS[@]}"; do
  declare -a args=(
    --benchmark_id="${benchmark_id}"
    --device="${DEVICE}"
    --output_path="${OUTPUT_PATH}"
    --iterations="${ITERATIONS}"
    --cache_dir="${CACHE_DIR}"
  )

  JAX_PLATFORMS="${DEVICE}" python3 "${TD}/benchmark_model.py" "${args[@]}"
done
