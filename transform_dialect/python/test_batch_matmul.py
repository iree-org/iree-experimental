# Instructions
#   Build IREE in `IREE_BUILD_DIR` with python bindings
#   source ${IREE_BUILD_DIR}/.env && export PYTHONPATH
#   export PATH=${PATH}:${IREE_BUILD_DIR}/tools
#
# Debug TD stuff by appending: --td-repro=1
#
# python test_batch_matmul.py

import matmul_config as config
import matmul_runner as runner
import td_argparse

problem_sizes = [
   [8, 128, 128, 128],
   [8, 128, 128, 130],
   [16, 64, 64, 256],
   [13, 70, 68, 259],
]
data_types = [
  ["f32", "f32", "f32"]
]

td_configurations = [
  {'blk': '4,4,32', 'tds': '32,4,4', 'wps': '1,4,4', 'p': 3, 'r': 32, 'acp': "1", 'mma': "0", 'wmma': "0", "fma": "1", "peel": 1},
  {'blk': '8,2,64', 'tds': '64,2,4', 'wps': '2,2,4', 'p': 1, 'r': 32, 'acp': "1", 'mma': "0", 'wmma': "0", "fma": "1", "peel": 1},
]

args = td_argparse.parse_args()

n_iters = 5
check_results = False
runner.run(
  lambda *args: config.make_fill_batch_matmul_problem(config.fill_batch_matmul_template, *args),
  problem_sizes, data_types, td_configurations, args, n_iters,
  runner.make_fill_batch_matmul_tensors,
  runner.torch_baseline_fill_batch_matmul_tensors,
  check_results)
