# /usr/local/cuda/bin/nsys profile --stats=true  python matmul.py 
# /usr/local/cuda/bin/nsys nvprof --print-gpu-trace python matmul.py 

import torch
torch.manual_seed(0)

import iree.runtime as ireert
from iree.runtime import get_driver, get_device
import iree.compiler as ireec
iree_device = "cuda"
iree_runtime_device = "cuda"

import compile_and_compare as cc
import matmul_config as config
import sys

ones_initialization_fn = lambda x, y: torch.ones(x, y)
linspace_initialization_fn = lambda x, y: torch.linspace(0, x*y, 1).reshape(x, y)
randn_initialization_fn = lambda x, y: torch.randn(x, y)
def make_fill_matmul_f32_tensors(M, N, K, initialization_fn=lambda x, y: torch.randn(x, y)):
  return initialization_fn(M, K), initialization_fn(K, N)

n_iters = 5
problem_sizes = [
  [514, 130, 500],
  [515, 132, 512],
  [515, 131, 501],
]

td_configurations = [
  # {'bx': 128, 'by': 128, 'bz': 1, 'tx': 64, 'ty': 2, 'tz': 1, 'wx': 2, 'wy': 2, 'wz': 1, 'pipe_depth': 2, 'red_sz': 16, 'async_cp': "false", 'mma_sync': "true"},
  # {'bx': 128, 'by': 128, 'bz': 1, 'tx': 64, 'ty': 2, 'tz': 1, 'wx': 2, 'wy': 2, 'wz': 1, 'pipe_depth': 3, 'red_sz': 16, 'async_cp': "true", 'mma_sync': "false"},
  # {'bx': 128, 'by': 128, 'bz': 1, 'tx': 64, 'ty': 2, 'tz': 1, 'wx': 2, 'wy': 2, 'wz': 1, 'pipe_depth': 4, 'red_sz': 16, 'async_cp': "true", 'mma_sync': "false"},
  # {'bx': 128, 'by': 128, 'bz': 1, 'tx': 64, 'ty': 2, 'tz': 1, 'wx': 2, 'wy': 2, 'wz': 1, 'pipe_depth': 5, 'red_sz': 16, 'async_cp': "true", 'mma_sync': "true"},
  {'bx': 16, 'by': 16, 'bz': 1, 'tx': 32, 'ty': 1, 'tz': 1, 'wx': 1, 'wy': 1, 'wz': 1, 'pipe_depth': 3, 'red_sz': 16, 'async_cp': "true", 'mma_sync': "true"},
]

for td_config in td_configurations:
  for M, N, K in problem_sizes:
    ir_str, fn_name = config.make_fill_matmul_f32_problem(M, N, K, td_config)
    lhs, rhs = make_fill_matmul_f32_tensors(M, N, K)
    print(ir_str)
    # print(f"td_config: {td_config}")

    td_vmfb = ireec.compile_str(
      ir_str,
      target_backends=[iree_device],
      extra_args=config.make_iree_td_options(td_config) + [
        # TODO: remove this if we can start IREE from cuda tensors.
        "--iree-hal-benchmark-dispatch-repeat-count=3",
      ]
    )
    td_result_0 = torch.from_numpy(
      cc.run_vmfb(td_vmfb, fn_name, iree_runtime_device, [lhs, rhs])).cuda()

    torch_result = torch.mm(lhs.cuda(), rhs.cuda())

