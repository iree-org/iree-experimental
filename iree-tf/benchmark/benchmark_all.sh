#!/bin/bash

DEVICE=$1
TENSORFLOW_VERSION=$2
OUTPUT_PATH=$3

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="tf-benchmarks.venv"

VENV_DIR="${VENV_DIR}" TENSORFLOW_VERSION="${TENSORFLOW_VERSION}" ${TD}/setup_venv.sh
source ${VENV_DIR}/bin/activate

MODEL_RESNET50_FP32_TF="2e1bd635-eeb3-41fa-90a6-e1cfdfa9be0a"
MODEL_BERT_LARGE_FP32_TF="979ff492-f363-4320-875f-e1ef93521132"
MODEL_T5_LARGE_FP32_TF="723da674-f42e-4d14-991e-16ad86a0d81b"

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
    --hlo_iterations="${ITERATIONS}"
  )

  if [ -z "${TF_RUN_HLO_MODULE_PATH}" ]; then
    echo "HLO Benchmark Path not set in environment variable TF_RUN_HLO_MODULE_PATH. Disabling compiler-level benchmarks."
  else
    args+=(
      --hlo_benchmark_path="${TF_RUN_HLO_MODULE_PATH}"
    )
  fi

  python "${TD}/benchmark_model.py" "${args[@]}"
done
