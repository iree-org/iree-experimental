#!/bin/bash

# Instructions
#   Build IREE in `IREE_BUILD_DIR` with python bindings
#   source ${IREE_BUILD_DIR}/.env && export PYTHONPATH
#   export PATH=${PATH}:${IREE_BUILD_DIR}/tools

# Debug TD stuff by appending: --td-repro=1

python matmul_test.py
python matmul_pad_test.py
python matmul_pad_test.py --td-graph-script=./td_scripts/matmul_pad.mlir
python matmul_pad_test.py --td-graph-script=./td_scripts/matmul_pad_split_k.mlir

/usr/local/cuda/bin/nsys profile --stats=true python matmul_bench.py
/usr/local/cuda/bin/nsys profile --stats=true python matmul_pad_bench.py
/usr/local/cuda/bin/nsys profile --stats=true python matmul_pad_bench.py --td-graph-script=./td_scripts/matmul_pad.mlir
/usr/local/cuda/bin/nsys profile --stats=true python matmul_pad_bench.py --td-graph-script=./td_scripts/matmul_pad_split_k.mlir
