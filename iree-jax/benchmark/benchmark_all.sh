#!/bin/bash

DEVICE=$1
OUTPUT_PATH=$2

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="jax-benchmarks.venv"

VENV_DIR="${VENV_DIR}" ${TD}/setup_venv.sh
source ${VENV_DIR}/bin/activate

MODEL_BERT_LARGE_FP32_JAX="f76dc3a5-3379-49ab-85e5-744ff5167310"

declare -a gpu_benchmark_ids=(
  "${MODEL_BERT_LARGE_FP32_JAX}-batch1"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch16"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch24"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch32"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch48"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch64"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch512"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch1024"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch1280"
)

declare -a cpu_benchmark_ids=(
  "${MODEL_BERT_LARGE_FP32_JAX}-batch1"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch32"
  "${MODEL_BERT_LARGE_FP32_JAX}-batch64"
)

if [ "${DEVICE}" = "gpu" ]; then
    BENCHMARK_IDS=("${gpu_benchmark_ids[@]}")
    ITERATIONS=5
else
    BENCHMARK_IDS=("${cpu_benchmark_ids[@]}")
    ITERATIONS=5
fi

# Create json file and populate with global information.
rm "${OUTPUT_PATH}"
echo "{\"trigger\": { \"timestamp\": \"$(date +'%s')\" }, \"benchmarks\": []}" > "${OUTPUT_PATH}"

for benchmark_id in "${BENCHMARK_IDS[@]}"; do
  declare -a args=(
    --benchmark_id="${benchmark_id}"
    --device="${DEVICE}"
    --output_path="${OUTPUT_PATH}"
    --iterations="${ITERATIONS}"
  )

  python "${TD}/benchmark_model.py" "${args[@]}"
done
