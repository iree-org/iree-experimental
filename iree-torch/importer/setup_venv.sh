#!/bin/bash

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="$TD/torch-models.venv"
if [ -z "$PYTHON" ]; then
  PYTHON="$(which python)"
fi

echo "Setting up venv dir: $VENV_DIR"
echo "Python: $PYTHON"
echo "Python version: $("$PYTHON" --version)"

function die() {
  echo "Error executing command: $*"
  exit 1
}

$PYTHON -m venv "$VENV_DIR" || die "Could not create venv."
source "$VENV_DIR/bin/activate" || die "Could not activate venv"

# Upgrade pip and install requirements. 'python' is used here in order to
# reference to the python executable from the venv.
python -m pip install --upgrade pip || die "Could not upgrade pip"
python -m pip install --upgrade -r "$TD/requirements.txt"

python -m pip install -f https://llvm.github.io/torch-mlir/package-index/ torch-mlir
python -m pip install -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html --pre torch torchvision accelerate

echo "Activate venv with:"
echo "  source $VENV_DIR/bin/activate"
