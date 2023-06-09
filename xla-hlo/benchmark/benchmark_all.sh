#!/bin/bash

DEVICE=$1
HLO_BENCHMARK_PATH=$2
OUTPUT_PATH=$3

MODEL_RESNET50_FP32_TF="aff75509-4420-40a8-844e-dbfc48494fe6-MODEL_RESNET50-fp32-TF-224x224x3xf32"
MODEL_BERT_LARGE_FP32_TF="47cb0d3a-5eb7-41c7-9d7c-97aae7023ecf-MODEL_BERT_LARGE-fp32-TF-384xi32"
MODEL_T5_LARGE_FP32_TF="173c7180-bad4-4b91-8423-4beeb13d2b0a-MODEL_T5_LARGE-fp32-TF-512xi32"
MODEL_RESNET50_FP32_JAX="aff75509-4420-40a8-844e-dbfc48494fe6-MODEL_RESNET50-fp32-JAX-3x224x224xf32"
MODEL_BERT_LARGE_FP32_JAX="47cb0d3a-5eb7-41c7-9d7c-97aae7023ecf-MODEL_BERT_LARGE-fp32-JAX-384xi32"
MODEL_T5_LARGE_FP32_JAX="173c7180-bad4-4b91-8423-4beeb13d2b0a-MODEL_T5_LARGE-fp32-JAX-512xi32"

declare -a gpu_benchmark_ids=(
  "${MODEL_RESNET50_FP32_TF}-batch1"
  "${MODEL_RESNET50_FP32_TF}-batch8"
  "${MODEL_RESNET50_FP32_TF}-batch64"
  "${MODEL_RESNET50_FP32_TF}-batch128"
  "${MODEL_RESNET50_FP32_TF}-batch256"
  "${MODEL_RESNET50_FP32_TF}-batch2048"
  "${MODEL_BERT_LARGE_FP32_TF}-batch1"
  "${MODEL_BERT_LARGE_FP32_TF}-batch16"
  "${MODEL_BERT_LARGE_FP32_TF}-batch24"
  "${MODEL_BERT_LARGE_FP32_TF}-batch32"
  "${MODEL_BERT_LARGE_FP32_TF}-batch48"
  "${MODEL_BERT_LARGE_FP32_TF}-batch64"
  "${MODEL_BERT_LARGE_FP32_TF}-batch512"
  "${MODEL_BERT_LARGE_FP32_TF}-batch1024"
  "${MODEL_BERT_LARGE_FP32_TF}-batch1280"
  "${MODEL_T5_LARGE_FP32_TF}-batch1"
  "${MODEL_T5_LARGE_FP32_TF}-batch16"
  "${MODEL_T5_LARGE_FP32_TF}-batch24"
  "${MODEL_T5_LARGE_FP32_TF}-batch32"
  "${MODEL_T5_LARGE_FP32_TF}-batch48"
  "${MODEL_T5_LARGE_FP32_TF}-batch64"
  "${MODEL_T5_LARGE_FP32_TF}-batch512"
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
  "${MODEL_RESNET50_FP32_TF}-batch1"
  "${MODEL_RESNET50_FP32_TF}-batch64"
  "${MODEL_RESNET50_FP32_TF}-batch128"
  "${MODEL_BERT_LARGE_FP32_TF}-batch1"
  "${MODEL_BERT_LARGE_FP32_TF}-batch32"
  "${MODEL_BERT_LARGE_FP32_TF}-batch64"
  "${MODEL_T5_LARGE_FP32_TF}-batch1"
  "${MODEL_T5_LARGE_FP32_TF}-batch16"
  "${MODEL_T5_LARGE_FP32_TF}-batch32"
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
    # Since each iteration includes both compilation and inference, we keep the
    # total iterations small because of the amount of time it takes to do both.
    # Std deviation is <1ms.
    BENCHMARK_IDS=("${cpu_benchmark_ids[@]}")
    ITERATIONS=5
fi

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="hlo-benchmarks.venv"

VENV_DIR="${VENV_DIR}" ${TD}/setup_venv.sh
source ${VENV_DIR}/bin/activate

CACHE_DIR="$(pwd)/.cache/oobi/models"
python "${TD}/download_artifacts.py" -o "${CACHE_DIR}" -bids "${BENCHMARK_IDS[@]}"

# Create json file and populate with global information.
rm "${OUTPUT_PATH}"
echo "{\"trigger\": { \"timestamp\": \"$(date +'%s')\" }, \"benchmarks\": [], \"execution_environment\": { \"xla_sha\": \"${XLA_SHA}\"}}" > "${OUTPUT_PATH}"

for benchmark_id in "${BENCHMARK_IDS[@]}"; do
  declare -a args=(
    --benchmark_id="${benchmark_id}"
    --device="${DEVICE}"
    --output_path="${OUTPUT_PATH}"
    --iterations="${ITERATIONS}"
    --hlo_benchmark_path="${HLO_BENCHMARK_PATH}"
    --cache_dir="${CACHE_DIR}"
  )

  python "${TD}/benchmark_model.py" "${args[@]}"
done
