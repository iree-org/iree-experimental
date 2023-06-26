# Instructions
#   Build IREE in `IREE_BUILD_DIR` with python bindings
#   source ${IREE_BUILD_DIR}/.env && export PYTHONPATH
#   export PATH=${PATH}:${IREE_BUILD_DIR}/tools
#
# Debug TD stuff by appending: --td-repro=1

# python matmul_pad_test.py
# python matmul_test.py --td-graph-script=./td_scripts/matmul_pad.mlir
# python matmul_test.py --td-graph-script=./td_scripts/matmul_pad_split_k.mlir

problem_sizes = [ 
  [123, 123, 12345] 
]
td_configurations = [
  {'blk': '16,16,1', 'tds': '32,1,1', 'wps': '1,1,1', 'p': 3, 'r': 16, 'acp': "1", 'mma': "1"},
]
data_types = [
  ["f32", "f32", "f32"]
]

import matmul_runner as runner
import td_argparse
args = td_argparse.parse_args()

n_iters = 5
check_results = True
runner.run(problem_sizes, data_types, td_configurations, args, n_iters, check_results = check_results)
