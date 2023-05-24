#!/bin/bash

# Sets up a virtual environment suitable for running `iree-tf`.
#
# Environment variables:
#   VENV_DIR=tf-benchmarks.venv
#   TENSORFLOW_VERSION=2.12.0

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR=${VENV_DIR:-tf-benchmarks.venv}

echo "Setting up venv dir: $VENV_DIR"

function die() {
  echo "Error executing command: $*"
  exit 1
}

python3 -m venv "${VENV_DIR}" || die "Could not create venv."
source "${VENV_DIR}/bin/activate" || die "Could not activate venv"

# Upgrade pip and install requirements. 'python' is used here in order to
# reference to the python executable from the venv.
python3 -m pip install --upgrade pip || die "Could not upgrade pip"

if [[ ! -z "${TENSORFLOW_VERSION}" ]]; then
  python3 -m pip install tensorflow==${TENSORFLOW_VERSION}
fi

# If the TF version is an release candidate, install the dev version of transformers.
if [[ "${TENSORFLOW_VERSION}" == *-rc* ]]; then
  python3 -m pip install --pre keras
  python3 -m pip install git+https://github.com/huggingface/transformers
else
  python3 -m pip install keras
  python3 -m pip install transformers
fi

python3 -m pip install -r "${TD}/requirements.txt"

echo "Activate venv with:"
echo "  source ${VENV_DIR}/bin/activate"
