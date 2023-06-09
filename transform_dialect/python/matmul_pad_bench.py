# Instructions
#   Build IREE in `IREE_BUILD_DIR` with python bindings
#   source ${IREE_BUILD_DIR}/.env && export PYTHONPATH
#   export PATH=${PATH}:${IREE_BUILD_DIR}/tools

# /usr/local/cuda/bin/nsys nvprof --print-gpu-trace python matmul_pad_bench.py
# /usr/local/cuda/bin/nsys nvprof --print-gpu-trace python matmul_pad_bench.py --td-graph-script=./td_scripts/matmul_pad.mlir
# /usr/local/cuda/bin/nsys nvprof --print-gpu-trace python matmul_pad_bench.py --td-graph-script=./td_scripts/matmul_pad_split_k.mlir
padding_problem_sizes = [
   [123, 456, 12345] 
]
padding_td_configurations = [
  {'blk': '16,16,1', 'tds': '32,1,1', 'wps': '1,1,1', 'p': 3, 'r': 16, 'acp': "1", 'mma': "1"},
]

import matmul_runner as runner
import td_argparse
args = td_argparse.parse_args()

n_iters = 1
runner.run(padding_problem_sizes, padding_td_configurations, args, n_iters)
