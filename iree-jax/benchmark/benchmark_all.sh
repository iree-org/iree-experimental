#!/bin/bash

DEVICE=$1
OUTPUT_PATH=$2

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="jax-benchmarks.venv"

VENV_DIR="${VENV_DIR}" ${TD}/setup_venv.sh
source ${VENV_DIR}/bin/activate

MODEL_RESNET50_FP32_JAX="aff75509-4420-40a8-844e-dbfc48494fe6-MODEL_RESNET50-fp32-JAX-3x224x224xf32"
MODEL_BERT_LARGE_FP32_JAX="47cb0d3a-5eb7-41c7-9d7c-97aae7023ecf-MODEL_BERT_LARGE-fp32-JAX-384xi32"
MODEL_T5_LARGE_FP32_JAX="173c7180-bad4-4b91-8423-4beeb13d2b0a-MODEL_T5_LARGE-fp32-JAX-512xi32"

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
    ITERATIONS=50
else
    BENCHMARK_IDS=("${cpu_benchmark_ids[@]}")
    ITERATIONS=20
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
