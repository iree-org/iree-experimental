#!/bin/bash

DEVICE=$1
TENSORFLOW_VERSION=$2
OUTPUT_PATH=$3
HLO_BENCHMARK_PATH=$4

VENV_DIR="tf-benchmarks.venv"

VENV_DIR="${VENV_DIR}" TENSORFLOW_VERSION="${TENSORFLOW_VERSION}" ./setup_venv.sh
source ${VENV_DIR}/bin/activate

MODEL_RESNET50_FP32_TF="2e1bd635-eeb3-41fa-90a6-e1cfdfa9be0a"
MODEL_BERT_LARGE_FP32_TF="979ff492-f363-4320-875f-e1ef93521132"
MODEL_T5_LARGE_FP32_TF="723da674-f42e-4d14-991e-16ad86a0d81b"

declare -a benchmark_ids=(
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
)

APPEND=false

for benchmark_id in "${benchmark_ids[@]}"; do
  declare -a args=(
    --benchmark_id="${benchmark_id}"
    --device=gpu
    --output_path="${OUTPUT_PATH}"
    --iterations=100
    --hlo_iterations=100
  )

  if ${APPEND}; then
    args+=(
      --append
    )
  fi

  if [ -z "${HLO_BENCHMARK_PATH}" ]; then
    echo "HLO Benchmark Path not provided. Disabling compiler-level benchmarks."
  else
    args+=(
      --hlo_benchmark_path="${HLO_BENCHMARK_PATH}"
    )
  fi

  python benchmark_model.py "${args[@]}"
  APPEND=true
done
