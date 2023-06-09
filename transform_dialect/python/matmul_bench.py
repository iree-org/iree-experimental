# Instructions
#   Build IREE in `IREE_BUILD_DIR` with python bindings
#   source ${IREE_BUILD_DIR}/.env && export PYTHONPATH
#   export PATH=${PATH}:${IREE_BUILD_DIR}/tools

# /usr/local/cuda/bin/nsys profile --stats=true  python matmul_bench.py 
# /usr/local/cuda/bin/nsys nvprof --print-gpu-trace python matmul_bench.py 
# sudo -E -- bash -c 'source ${IREE_BUILD_DIR}/.env && export PYTHONPATH && /usr/local/cuda/bin/ncu -f --set full -o profile python matmul_bench.py'

problem_sizes = [
  [514, 130, 500],
  [515, 132, 512],
  [515, 131, 501],
  [1020, 1020, 1020],
  [1024, 1024, 1024],
  [1920, 1920, 1920],
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

n_iters = 1
runner.run(problem_sizes, td_configurations, args, n_iters)
