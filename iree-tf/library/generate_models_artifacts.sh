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

pip list > ${OUTPUT_DIR}/version_info.txt

python generate_saved_models.py -o ${OUTPUT_DIR}

# Zip the saved models.
MODEL_DIRS=$(find "${OUTPUT_DIR}/" -maxdepth 1 -mindepth 1 -type d)

for model_dir in ${MODEL_DIRS}; do
    BATCH_DIRS=$(find "${model_dir}/" -maxdepth 1 -mindepth 1 -type d)
    for batch_dir in ${BATCH_DIRS}; do
        SM_DIR="${batch_dir}/saved_model"
        pushd "${SM_DIR}"
        tar -czvf ../tf-model.tar.gz .
        popd
        # Delete the unzipped directory.
        rm -rf "${SM_DIR}"
    done
done

echo "SavedModel generation complete. Please upload ${OUTPUT_DIR} to gs://iree-model-artifacts/tensorflow."
