# Instructions
#   Build IREE in `IREE_BUILD_DIR` with python bindings
#   source ${IREE_BUILD_DIR}/.env && export PYTHONPATH
#   export PATH=${PATH}:${IREE_BUILD_DIR}/tools
#
# Debug TD stuff by appending: --td-repro=1

# python test_matmul.py

problem_sizes = [
  # Partially aligned
  [133, 133, 128],
  [167, 171, 128],
  [167, 170, 128],
  [167, 172, 128],
  [330, 330, 128],
  [512, 132, 515],
  [515, 128, 513],
  [515, 130, 512],
  [515, 133, 512],
  [516, 130, 512],
  [516, 131, 512],

  # Unaligned
  [514, 130, 500],
  [515, 131, 501],

  # Fully aligned
  [512, 256, 128],
  [512, 512, 512],
]
td_configurations = [
  {'blk': '16,16,1', 'tds': '32,1,1', 'wps': '1,1,1', 'p': 1, 'r': 16, 'acp': "0", 'mma': "1", 'wmma': "0", "fma": "0"},
  {'blk': '16,16,1', 'tds': '32,1,1', 'wps': '1,1,1', 'p': 1, 'r': 16, 'acp': "1", 'mma': "1", 'wmma': "0", "fma": "0"},
  {'blk': '16,16,1', 'tds': '32,1,1', 'wps': '1,1,1', 'p': 1, 'r': 16, 'acp': "0", 'mma': "0", 'wmma': "1", "fma": "0"},
  {'blk': '16,16,1', 'tds': '32,1,1', 'wps': '1,1,1', 'p': 1, 'r': 16, 'acp': "1", 'mma': "0", 'wmma': "1", "fma": "0"},
  {'blk': '16,16,1', 'tds': '32,1,1', 'wps': '1,1,1', 'p': 1, 'r': 16, 'acp': "0", 'mma': "0", 'wmma': "0", "fma": "1"},
  {'blk': '16,16,1', 'tds': '32,1,1', 'wps': '1,1,1', 'p': 1, 'r': 16, 'acp': "1", 'mma': "0", 'wmma': "0", "fma": "1"},

  {'blk': '128,128,1', 'tds': '64,2,1', 'wps': '2,2,1', 'p': 1, 'r': 16, 'acp': "0", 'mma': "0", 'wmma': "0", "fma": "1"},
  {'blk': '128,128,1', 'tds': '64,2,1', 'wps': '2,2,1', 'p': 1, 'r': 16, 'acp': "1", 'mma': "1", 'wmma': "0", "fma": "0"},
  {'blk': '128,128,1', 'tds': '64,2,1', 'wps': '2,2,1', 'p': 3, 'r': 16, 'acp': "1", 'mma': "0", 'wmma': "1", "fma": "0"},
  {'blk': '128,128,1', 'tds': '64,2,1', 'wps': '2,2,1', 'p': 5, 'r': 16, 'acp': "1", 'mma': "1", 'wmma': "0", "fma": "0"},
  {'blk': '32,32,1', 'tds': '32,1,1', 'wps': '1,1,1', 'p': 1, 'r': 16, 'acp': "0", 'mma': "1", 'wmma': "0", "fma": "0"},
  {'blk': '32,32,1', 'tds': '64,1,1', 'wps': '2,1,1', 'p': 3, 'r': 16, 'acp': "1", 'mma': "0", 'wmma': "0", "fma": "1"},
  {'blk': '32,32,1', 'tds': '64,1,1', 'wps': '1,2,1', 'p': 3, 'r': 16, 'acp': "1", 'mma': "1", 'wmma': "0", "fma": "0"},
  {'blk': '16,16,1', 'tds': '32,1,1', 'wps': '1,1,1', 'p': 3, 'r': 16, 'acp': "1", 'mma': "0", 'wmma': "0", "fma": "1"},
  {'blk': '16,16,1', 'tds': '32,1,1', 'wps': '1,1,1', 'p': 7, 'r': 16, 'acp': "1", 'mma': "1", 'wmma': "0", "fma": "0"},

  {'blk': '128,128,1', 'tds': '64,2,1', 'wps': '2,2,1', 'p': 1, 'r': 16, 'acp': "0", 'mma': "0", 'wmma': "1", "fma": "0"},
  {'blk': '128,128,1', 'tds': '64,2,1', 'wps': '2,2,1', 'p': 3, 'r': 16, 'acp': "1", 'mma': "0", 'wmma': "1", "fma": "0"},
  {'blk': '32,32,1', 'tds': '64,1,1', 'wps': '2,1,1', 'p': 3, 'r': 16, 'acp': "1", 'mma': "0", 'wmma': "1", "fma": "0"},
  {'blk': '16,16,1', 'tds': '32,1,1', 'wps': '1,1,1', 'p': 3, 'r': 16, 'acp': "1", 'mma': "0", 'wmma': "1", "fma": "0"},
]
data_types = [
  ["f32", "f32", "f32"]
]

import matmul_config as config
import matmul_runner as runner
import td_argparse
args = td_argparse.parse_args()

n_iters = 5
check_results = True
runner.run(
  lambda *args: config.make_fill_matmul_problem(config.fill_matmul_template, *args),
  problem_sizes, data_types, td_configurations, args, n_iters, 
  tensor_builder_fn = runner.make_fill_matmul_tensors,
  torch_baseline_fn = runner.torch_baseline_fill_matmul_tensors,
  check_results = check_results)

# transpose_a does not correctly generate the mma sync instructions and results
# in extremely long compile + runtimes atm.
# runner.run(problem_sizes, data_types, td_configurations, args, n_iters, \
#            template_str=config.fill_matmul_transpose_a_template, \
#            tensor_builder_fn=runner.make_fill_matmul_transpose_a_tensors, \
#            torch_baseline_fn=runner.torch_baseline_fill_matmul_transpose_a_tensors, \
#            check_results = check_results)

# TODO: transpose_b works fine but needs to be matched properly in IREE.
# runner.run(problem_sizes, data_types, td_configurations, args, n_iters, \
#            template_str=config.fill_matmul_transpose_b_template, \
#            tensor_builder_fn=runner.make_fill_matmul_transpose_b_tensors, \
#            torch_baseline_fn=runner.torch_baseline_fill_matmul_transpose_b_tensors, \
#            check_results = check_results)
