#!/bin/bash

# Runs `generate_model_artifacts.py` on all registered JAX models and saves all artifacts in to the directory `/tmp/jax_models_<jax-version>_<timestamp>`.
# Once complete. please upload the output directory to `gs://iree-model-artifacts/jax`, preserving directory name.
#
# Usage:
#     bash generate_saved_models.sh <path_to_iree_opt>
#
# Requires python-3.10 and above and python-venv.

IREE_OPT_PATH=$1

rm -rf jax-models.venv
bash setup_venv.sh
source jax-models.venv/bin/activate

JAX_VERSION=$(pip show jax | grep Version | sed -e "s/^Version: \(.*\)$/\1/g")
DIR_NAME="jax_models_${JAX_VERSION}_$(date +'%s')"
OUTPUT_DIR="/tmp/${DIR_NAME}"
mkdir "${OUTPUT_DIR}"

pip list > "${OUTPUT_DIR}/models_version_info.txt"

python generate_model_artifacts.py -o "${OUTPUT_DIR}"

pushd ${OUTPUT_DIR}
for model_dir in */; do
    pushd "${model_dir}"
    for batch_dir in */; do
        pushd "${batch_dir}"

        # Binarize mlir artifacts.
        ${IREE_OPT_PATH} --emit-bytecode "stablehlo.mlir" -o "stablehlo.mlirbc"
        rm "stablehlo.mlir"

        # The name of the input file varies depending on the number of modules compiled.
        # Copy the file to a name that is known and static.
        HLO_INPUT_PATH=$(realpath "hlo/*.jit_forward.before_optimizations.txt")
        cp ${HLO_INPUT_PATH} "hlo/jit_forward.before_optimizations.txt"

        # Remove compiled hlo artifacts since we only need the input hlo.
        rm hlo/*.ll
        rm hlo/*.o

        popd
    done
    popd
done
popd

echo "Model artifact generation complete. Please upload ${OUTPUT_DIR} to gs://iree-model-artifacts/jax."
