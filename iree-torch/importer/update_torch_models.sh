#!/bin/bash

# Runs `torch_importer` and uploads generated artifacts to gcs.

rm -rf torch-models.venv
bash setup_venv.sh
source torch-models.venv/bin/activate

TORCH_MLIR_VERSION=$(pip show torch-mlir | grep Version | sed -e "s/^Version: \(.*\)$/\1/g")
DIR_NAME="torch_models_${TORCH_MLIR_VERSION}_$(date +'%s')"
OUTPUT_DIR="/tmp/${DIR_NAME}"
mkdir ${OUTPUT_DIR}

pip list > ${OUTPUT_DIR}/version_info.txt

python3 torch_importer.py -o ${OUTPUT_DIR}

GCS_DIR="gs://iree-model-artifacts/pytorch/${DIR_NAME}"
gcloud storage cp "${OUTPUT_DIR}/**" "${GCS_DIR}/"
