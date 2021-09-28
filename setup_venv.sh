#!/bin/bash
# Sets up a venv suitable for running samples.
# Recommend getting default 'python' to be python 3. For example on Debian:
#   sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1
# Or launch with python=/some/path
td="$(cd $(dirname $0) && pwd)"
venv_dir="$td/iree-samples.venv"
if [ -z "$python" ]; then
  python="$(which python)"
fi

echo "Setting up venv dir: $venv_dir"
echo "Python: $python"
echo "Python version: $("$python" --version)"

function die() {
  echo "Error executing command: $*"
  exit 1
}

python -m venv "$venv_dir" || die "Could not create venv."
source "$venv_dir/bin/activate" || die "Could not activate venv"

# Upgrade pip.
python -m pip install --upgrade pip || die "Could not upgrade pip"

# Install local binaries.
python -m pip install \
  -f https://github.com/google/iree/releases --upgrade \
  iree-compiler-snapshot \
  iree-runtime-snapshot \
  iree-tools-tflite-snapshot \
  iree-tools-xla-snapshot \
  iree-tools-tf-snapshot \
  || die "Could not install IREE deps"

# Install dependencies.
python -m pip install --upgrade tensorflow || die "Could not install TensorFlow"
python -m pip install --upgrade "jax[cpu]" || die "Could not install JAX"

# tflitehub dependencies.
python -m pip install Pillow || die "Could not install Pillow"

echo "Activate venv with:"
echo "  source $venv_dir/bin/activate"
