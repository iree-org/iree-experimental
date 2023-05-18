#!/bin/bash

DEVICE=$1
OUTPUT_PATH=$2

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="jax-benchmarks.venv"

VENV_DIR="${VENV_DIR}" ${TD}/setup_venv.sh
source ${VENV_DIR}/bin/activate

MODEL_RESNET50_FP32_JAX="c0a738bc-0c21-40b6-b565-31fe7fd33d0d"
MODEL_BERT_LARGE_FP32_JAX="f76dc3a5-3379-49ab-85e5-744ff5167310"
MODEL_T5_LARGE_FP32_JAX="7720beef-ac1a-4a5f-8777-505ea949a138"

declare -a gpu_benchmark_ids=(
  "${MODEL_RESNET50_FP32_JAX}-batch1"
  "${MODEL_RESNET50_FP32_JAX}-batch8"
  "${MODEL_RESNET50_FP32_JAX}-batch64"
  "${MODEL_RESNET50_FP32_JAX}-batch128"
  "${MODEL_RESNET50_FP32_JAX}-batch256"
  "${MODEL_RESNET50_FP32_JAX}-batch2048"
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
