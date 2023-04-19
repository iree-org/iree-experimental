#!/bin/bash

bash setup_venv.sh
source tf-benchmarks.venv/bin/activate

MODEL_RESNET50_FP32_TF="2e1bd635-eeb3-41fa-90a6-e1cfdfa9be0a"
BENCHMARKS_IDS=(
    "${MODEL_RESNET50_FP32_TF}-batch1"
    "${MODEL_RESNET50_FP32_TF}-batch8"
    "${MODEL_RESNET50_FP32_TF}-batch64"
    "${MODEL_RESNET50_FP32_TF}-batch128"
    "${MODEL_RESNET50_FP32_TF}-batch256"
    "${MODEL_RESNET50_FP32_TF}-batch2048"
)

OUTPUT_PATH="/tmp/tf_benchmarks.csv"
APPEND=false
for benchmark_id in ${BENCHMARKS_IDS[@]}; do
    python benchmark_model.py --benchmark_id ${benchmark_id} --device cpu --output_path ${OUTPUT_PATH} --append ${APPEND}
    APPEND=true
done
