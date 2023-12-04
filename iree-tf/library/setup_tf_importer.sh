#!/bin/bash

TD="$(cd $(dirname $0) && pwd)"
VENV_DIR="$TD/tf-importer.venv"
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
python -m pip install iree-tools-tf -f https://openxla.github.io/iree/pip-release-links.html
python -m pip install tf-nightly

echo "Activate venv with:"
echo "  source $VENV_DIR/bin/activate"
