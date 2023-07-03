# Instructions
#   Build IREE in `IREE_BUILD_DIR` with python bindings
#   source ${IREE_BUILD_DIR}/.env && export PYTHONPATH
#   export PATH=${PATH}:${IREE_BUILD_DIR}/tools

# /usr/local/cuda/bin/nsys profile --stats=true  python bench_batch_matmul.py --problem-size 8 4 64 128
# /usr/local/cuda/bin/nsys nvprof --print-gpu-trace python bench_batch_matmul.py  --problem-size 8 4 64 128
# sudo -E -- bash -c 'source ${IREE_BUILD_DIR}/.env && export PYTHONPATH && /usr/local/cuda/bin/ncu -f --set full -o profile python bench_batch_matmul.py --problem-size 8 4 64 128'

import matmul_config as config
import matmul_runner as runner
import td_argparse

args = td_argparse.parse_args()

if args.problem_size is None or len(args.problem_size) != 4:
  raise RuntimeError("expected 4 problem sizes")

problem_sizes = [args.problem_size]

data_types = [
  ["f32", "f32", "f32"]
]

td_configurations = [
  {'blk': '4,4,32', 'tds': '32,4,4', 'wps': '1,4,4', 'p': 3, 'r': 32, 'acp': "1", 'mma': "0", 'wmma': "0", "fma": "1"},
  {'blk': '8,2,64', 'tds': '64,2,4', 'wps': '2,2,4', 'p': 1, 'r': 32, 'acp': "1", 'mma': "0", 'wmma': "0", "fma": "1"},

  # WMMA is not supported for batch matmul, so this will fail to match and run as baseline.
  {'blk': '4,4,32', 'tds': '32,4,4', 'wps': '1,4,4', 'p': 1, 'r': 16, 'acp': "1", 'mma': "0", 'wmma': "1", "fma": "0"},
]

# We need at least two iterations, the first one includes data copies from IREE.
n_iters = 2
check_results = False
runner.run(
  lambda *args: config.make_fill_batch_matmul_problem(config.fill_batch_matmul_template, *args),
  problem_sizes, data_types, td_configurations, args, n_iters,
  runner.make_fill_batch_matmul_tensors,
  runner.torch_baseline_fill_batch_matmul_tensors,
  check_results)
