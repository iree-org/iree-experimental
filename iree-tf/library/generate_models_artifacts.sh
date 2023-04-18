#!/bin/bash

# Runs `generate_saved_models.py` on all registered Tensorflow models and saves all artifacts in to the directory `/tmp/tf_models_<tensorflow-version>_<timestamp>`.
# Once complete. please upload the output directory to `gs://iree-model-artifacts/tensorflow`, preserving directory name.
#
# Usage:
#     bash generate_saved_models.sh
#
# Requires python-3.10 and above and python-venv.

rm -rf tf-models.venv
bash setup_venv.sh
source tf-models.venv/bin/activate

TF_VERSION=$(pip show tensorflow | grep Version | sed -e "s/^Version: \(.*\)$/\1/g")
DIR_NAME="tf_models_${TF_VERSION}_$(date +'%s')"
OUTPUT_DIR="/tmp/${DIR_NAME}"
mkdir ${OUTPUT_DIR}

pip list > ${OUTPUT_DIR}/models_version_info.txt

python generate_models_artifacts.py -o ${OUTPUT_DIR}

# Generate hlo.
# We use a different python environment because we require the latest version of Tensorflow.
rm -rf tf-importer.venv
bash setup_tf_importer.sh
source tf-importer.venv/bin/activate

pip list > ${OUTPUT_DIR}/importer_version_info.txt

MODEL_DIRS=$(find "${OUTPUT_DIR}/" -maxdepth 1 -mindepth 1 -type d)
for model_dir in ${MODEL_DIRS}; do
    BATCH_DIRS=$(find "${model_dir}/" -maxdepth 1 -mindepth 1 -type d)
    for batch_dir in ${BATCH_DIRS}; do
        SM_DIR="${batch_dir}/saved_model"
        echo "Importing ${SM_DIR}"
        iree-import-tf --output-format=mlir-bytecode --tf-import-type=savedmodel_v2 --tf-savedmodel-exported-names=forward ${SM_DIR} -o "${batch_dir}/hlo.mlirbc"
    done
done

# Zip the saved models and hlo.
for model_dir in ${MODEL_DIRS}; do
    BATCH_DIRS=$(find "${model_dir}/" -maxdepth 1 -mindepth 1 -type d)
    for batch_dir in ${BATCH_DIRS}; do
        # Zip saved models.
        SM_DIR="${batch_dir}/saved_model"
        pushd "${SM_DIR}"
        tar -czvf ../tf-model.tar.gz .
        popd
        rm -rf "${SM_DIR}"
        # Zip hlo dump.
        HLO_DIR="${batch_dir}/hlo"
        pushd "${HLO_DIR}"
        tar -czvf ../hlo.tar.gz .
        popd
        rm -rf "${HLO_DIR}"
    done
done

echo "SavedModel generation complete. Please upload ${OUTPUT_DIR} to gs://iree-model-artifacts/tensorflow."
