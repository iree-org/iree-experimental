#!/bin/bash

# Instructions
#   Build IREE in `IREE_BUILD_DIR` with python bindings
#   source ${IREE_BUILD_DIR}/.env && export PYTHONPATH
#   export PATH=${PATH}:${IREE_BUILD_DIR}/tools

# Debug TD stuff by appending: --td-repro=1

python test_matmul_mixed.py
python test_matmul_pad.py
python test_matmul_transpose_a_mixed.py
python test_matmul_transpose_b_mixed.py
# The scripts are broken atm, we should move to nvgpu anyway.
# python test_matmul_pad.py --td-graph-script=./td_scripts/matmul_pad.mlir
#python test_matmul_pad.py --td-graph-script=./td_scripts/matmul_pad_split_k.mlir
python test_matmul.py

/usr/local/cuda/bin/nsys profile --stats=true python bench_matmul.py
/usr/local/cuda/bin/nsys profile --stats=true python bench_matmul_pad.py
# The scripts are broken atm, we should move to nvgpu anyway.
# /usr/local/cuda/bin/nsys profile --stats=true python bench_matmul_pad.py --td-graph-script=./td_scripts/matmul_pad.mlir
# /usr/local/cuda/bin/nsys profile --stats=true python bench_matmul_pad.py --td-graph-script=./td_scripts/matmul_pad_split_k.mlir
