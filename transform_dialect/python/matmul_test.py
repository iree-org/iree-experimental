# Instructions
#   Build IREE in `IREE_BUILD_DIR` with python bindings
#   source ${IREE_BUILD_DIR}/.env && export PYTHONPATH
#   export PATH=${PATH}:${IREE_BUILD_DIR}/tools
#
# Debug TD stuff by appending: --td-repro=1

# python matmul_test.py

problem_sizes = [
  # Partially aligned
  [133, 133, 128],
  [167, 171, 128],
  [167, 196, 128],
  [167, 197, 128],
  [167, 170, 128],
  [167, 172, 128],
  [167, 130, 128],
  [330, 330, 128],
  [512, 132, 515],
  [515, 128, 513],
  [515, 130, 512],
  [515, 131, 512],
  [515, 132, 512],
  [515, 133, 512],
  [516, 130, 512],
  [516, 131, 512],
  [516, 132, 512],
  [516, 133, 512],

  # Unaligned
  [514, 130, 500],
  [512, 512, 512],
  [515, 131, 501],

  # Fully aligned
  [512, 256, 128],
  [512, 512, 512],
  [1024, 1024, 1024],
]

td_configurations = [
  {'blk': '128,128,1', 'tds': '64,2,1', 'wps': '2,2,1', 'p': 1, 'r': 16, 'acp': "0", 'mma': "0"},
  {'blk': '128,128,1', 'tds': '64,2,1', 'wps': '2,2,1', 'p': 1, 'r': 16, 'acp': "1", 'mma': "1"},
  {'blk': '128,128,1', 'tds': '64,2,1', 'wps': '2,2,1', 'p': 3, 'r': 16, 'acp': "1", 'mma': "0"},
  {'blk': '128,128,1', 'tds': '64,2,1', 'wps': '2,2,1', 'p': 5, 'r': 16, 'acp': "1", 'mma': "1"},
  {'blk': '32,32,1', 'tds': '32,1,1', 'wps': '1,1,1', 'p': 1, 'r': 16, 'acp': "0", 'mma': "1"},
  {'blk': '32,32,1', 'tds': '64,1,1', 'wps': '2,1,1', 'p': 3, 'r': 16, 'acp': "1", 'mma': "1"},
  {'blk': '32,32,1', 'tds': '64,1,1', 'wps': '1,2,1', 'p': 3, 'r': 16, 'acp': "1", 'mma': "1"},
  {'blk': '16,16,1', 'tds': '32,1,1', 'wps': '1,1,1', 'p': 3, 'r': 16, 'acp': "1", 'mma': "0"},
  {'blk': '16,16,1', 'tds': '32,1,1', 'wps': '1,1,1', 'p': 7, 'r': 16, 'acp': "1", 'mma': "1"},
]

import matmul_runner as runner
import td_argparse
args = td_argparse.parse_args()

n_iters = 5
check_results = True
runner.run(problem_sizes, td_configurations, args, n_iters, check_results)
