# Torch frontend for IREE.

This is currently an experimental staging ground for using
[Torch-MLIR](https://github.com/llvm/torch-mlir) as a frontend for IREE. It is
expected to graduate to its own project/repository in the future as it matures.

Setup the venv for running:
```
# Install torch-mlir dependencies:
(iree-samples.venv) $ pip install --pre --upgrade torch torchvision -f https://download.pytorch.org/whl/nightly/cpu/torch_nightly.html
# Set up PYTHONPATH pointing at a built torch-mlir:
(iree-samples.venv) $ export PYTHONPATH="${PYTHONPATH}:${TORCHMLIR_SRC_ROOT}/build/tools/torch-mlir/python_packages/torch_mlir"
```

Run the torch-mlir e2e test suite for TorchScript:
```
(iree-samples.venv) $ "${TORCHMLIR_SRC_ROOT}/tools/torchscript_e2e_test.sh" -c external --external-config "${IREE_SAMPLES_SRC_ROOT}/torch-iree/torchscript_e2e_config.py"
```
