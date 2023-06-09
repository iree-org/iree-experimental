#!/bin/bash

DEVICE=$1
OUTPUT_PATH=$2

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="pt-benchmarks.venv"

VENV_DIR="${VENV_DIR}" ${TD}/setup_venv.sh
source ${VENV_DIR}/bin/activate

MODEL_RESNET50_FP32_PT="aff75509-4420-40a8-844e-dbfc48494fe6-MODEL_RESNET50-fp32-PT-3x224x224xf32"
MODEL_BERT_LARGE_FP32_PT="47cb0d3a-5eb7-41c7-9d7c-97aae7023ecf-MODEL_BERT_LARGE-fp32-PT-384xi32"

declare -a gpu_benchmark_ids=(
  "${MODEL_RESNET50_FP32_PT}-batch1"
  "${MODEL_RESNET50_FP32_PT}-batch8"
  "${MODEL_RESNET50_FP32_PT}-batch64"
  "${MODEL_RESNET50_FP32_PT}-batch128"
  "${MODEL_RESNET50_FP32_PT}-batch256"
  "${MODEL_RESNET50_FP32_PT}-batch2048"
  "${MODEL_BERT_LARGE_FP32_PT}-batch1"
  "${MODEL_BERT_LARGE_FP32_PT}-batch16"
  "${MODEL_BERT_LARGE_FP32_PT}-batch24"
  "${MODEL_BERT_LARGE_FP32_PT}-batch32"
  "${MODEL_BERT_LARGE_FP32_PT}-batch48"
  "${MODEL_BERT_LARGE_FP32_PT}-batch64"
  "${MODEL_BERT_LARGE_FP32_PT}-batch512"
  "${MODEL_BERT_LARGE_FP32_PT}-batch1024"
  "${MODEL_BERT_LARGE_FP32_PT}-batch1280"
)

declare -a cpu_benchmark_ids=(
  "${MODEL_RESNET50_FP32_PT}-batch1"
  "${MODEL_RESNET50_FP32_PT}-batch64"
  "${MODEL_RESNET50_FP32_PT}-batch128"
  "${MODEL_BERT_LARGE_FP32_PT}-batch1"
  "${MODEL_BERT_LARGE_FP32_PT}-batch32"
  "${MODEL_BERT_LARGE_FP32_PT}-batch64"
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
