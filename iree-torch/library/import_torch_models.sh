#!/bin/bash

# Runs `torch_importer` on all registered PyTorch models and saves the imported mlir files to the directory `/tmp/torch_models_<torch-mlir-version>_<timestamp>`.
# Once complete. please upload the output directory to `gs://iree-model-artifacts/pytorch`, preserving directory name.
#
# Usage:
#     bash update_torch_models.sh
#
# Requires python-3.10 and above and python-venv.
# Downloads and installs the latest `torch-mlir` and `torch` dev nightly.
#
# Note: Some models require CUDA. If no GPU is available, these models will be skipped.

rm -rf torch-models.venv
export WITH_CUDA=1
./setup_venv.sh
unset WITH_CUDA
source torch-models.venv/bin/activate

TORCH_MLIR_VERSION=$(pip show torch-mlir | grep Version | sed -e "s/^Version: \(.*\)$/\1/g")
DIR_NAME="torch_models_${TORCH_MLIR_VERSION}_$(date +'%s')"
OUTPUT_DIR="/tmp/${DIR_NAME}"
mkdir ${OUTPUT_DIR}

pip list > ${OUTPUT_DIR}/version_info.txt

python import_models.py -o ${OUTPUT_DIR}
