#!/bin/bash

# Sets up a virtual environment suitable for running `xla-hlo`.
#
# Environment variables:
#   VENV_DIR=hlo-benchmarks.venv

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR=${VENV_DIR:-hlo-benchmarks.venv}

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
python3 -m pip install -r "${TD}/requirements.txt"

echo "Activate venv with:"
echo "  source ${VENV_DIR}/bin/activate"
