#!/bin/bash

bash setup_venv.sh
source tf-benchmarks.venv/bin/activate

HLO_BENCHMARK_PATH="$1"

MODEL_RESNET50_FP32_TF="2e1bd635-eeb3-41fa-90a6-e1cfdfa9be0a"
declare -a benchmark_ids=(
  "${MODEL_RESNET50_FP32_TF}-batch1"
  "${MODEL_RESNET50_FP32_TF}-batch8"
  "${MODEL_RESNET50_FP32_TF}-batch64"
  "${MODEL_RESNET50_FP32_TF}-batch128"
  "${MODEL_RESNET50_FP32_TF}-batch256"
  "${MODEL_RESNET50_FP32_TF}-batch2048"
)

OUTPUT_PATH="/tmp/tf_benchmarks.csv"
APPEND=false

for benchmark_id in "${benchmark_ids[@]}"; do
  declare -a args=(
    --benchmark_id="${benchmark_id}"
    --device=gpu
    --output_path="${OUTPUT_PATH}"
    --hlo_benchmark_path="${HLO_BENCHMARK_PATH}"
    --iterations=100
    --hlo_iterations=100
  )

  if ${APPEND}; then
    args+=(
      --append
    )
  fi

  python benchmark_model.py "${args[@]}"
  APPEND=true
done
