# Instructions
#   Build IREE in `IREE_BUILD_DIR` with python bindings
#   source ${IREE_BUILD_DIR}/.env && export PYTHONPATH
#   export PATH=${PATH}:${IREE_BUILD_DIR}/tools
#
# Debug TD stuff by appending: --td-repro=1

# python test_matmul_transpose_b_mixed.py

problem_sizes = [
  # Partially aligned
  [123, 456, 789],
]
td_configurations = [
  {'blk': '16,16,1', 'tds': '32,1,1', 'wps': '1,1,1', 'p': 1, 'r': 16, 'acp': "1", 'mma': "0", 'wmma': "0", "fma": "1"},
]
data_types = [
  ["f16", "i8", "f16"],
  ["f16", "f16", "i8"],
]

import matmul_config as config
import matmul_runner as runner
import td_argparse
args = td_argparse.parse_args()

n_iters = 5
check_results = True
runner.run(problem_sizes, data_types, td_configurations, args, n_iters, \
           template_str=config.fill_matmul_transpose_b_template, \
           tensor_builder_fn=runner.make_fill_matmul_transpose_b_tensors, \
           torch_baseline_fn=runner.torch_baseline_fill_matmul_transpose_b_tensors, \
           check_results = check_results)
