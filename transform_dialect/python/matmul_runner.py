import torch
torch.manual_seed(0)

import iree.compiler as ireec
import iree.runtime as ireert
from iree.runtime import get_driver, get_device
iree_device = "cuda"
iree_runtime_device_name = "cuda"
iree_runtime_device = get_device(iree_runtime_device_name)

import compile_and_compare as cc
import matmul_config as config
import sys

ones_initialization_fn = lambda x, y: torch.ones(x, y)
linspace_initialization_fn = lambda x, y: torch.linspace(0, x*y - 1, x*y).reshape(x, y)
randn_initialization_fn = lambda x, y: torch.randn(x, y)

# General Qs to avoid working around the system:
#   * why does IREE have asymmetric APIs such as tensor.to_host() but ireert.asdevicearray(tensor) ?
#   * why not have something to build an IREE device tensor from a torch cuda tensor ?
#     - do we really need all these genuflexions through the host and maintaining 3 array copies ?
#   * either I am misusing this API or ireert.asdevicearray does not work as expected:
#     - the kernel cost as shown in nvprof is still heavily skewed by the copy cost
#     - it is still necessary to run multiple times to show the real timings
def make_fill_matmul_f32_tensors(M, N, K, initialization_fn=lambda x, y: torch.randn(x, y)):
  lhs, rhs = initialization_fn(M, K), initialization_fn(K, N)
  return lhs.cuda(), \
         rhs.cuda(), \
         ireert.asdevicearray(iree_runtime_device, lhs, implicit_host_transfer=False), \
         ireert.asdevicearray(iree_runtime_device, rhs, implicit_host_transfer=False)

def run(problem_sizes, td_configurations, argparse, n_iters, check_results=False):
  for M, N, K in problem_sizes:
    # init_fn = ones_initialization_fn
    # init_fn = linspace_initialization_fn
    init_fn = randn_initialization_fn
    ir_str, fn_name = config.make_fill_matmul_f32_problem(M, N, K)
    print(f"matmul_{M}x{N}x{K}")
    for td_config in td_configurations:
      print(f"td_config: {td_config}")
      td_ir_str, td_fn_name = config.make_fill_matmul_f32_problem(M, N, K, td_config)
      lhs_torch, rhs_torch, lhs_iree, rhs_iree = \
        make_fill_matmul_f32_tensors(M, N, K, init_fn)

      # Compile and run the baseline.
      extra_args=config.append_td_graph_script(
        config.make_iree_baseline_options(argparse.td_repro), \
        argparse.td_graph_script)
      # print(f"compile baseline {extra_args}")
      baseline_vmfb_str = ireec.compile_str(
        ir_str,
        target_backends=[iree_device],
        extra_args=extra_args,
      )
      # print("prepare baseline fun")
      baseline_fun = cc.prepare_fun(baseline_vmfb_str, fn_name, iree_runtime_device_name)
      # print("run baseline fun")
      baseline_result_0 = torch.from_numpy(baseline_fun(lhs_iree, rhs_iree).to_host())

      # Compile and run with TD options.
      extra_args = config.append_td_graph_script(
          config.make_iree_td_options(td_config, \
                                      argparse.td_repro), \
          argparse.td_graph_script)
      # print(f"compile td {extra_args}")
      td_vmfb_str = ireec.compile_str(
        td_ir_str,
        target_backends=[iree_device],
        extra_args=extra_args,
      )
      # print("prepare td fun")
      td_fun = cc.prepare_fun(td_vmfb_str, td_fn_name, iree_runtime_device_name)
      # print("run td fun")
      td_result_0 = torch.from_numpy(td_fun(lhs_iree, rhs_iree).to_host())

      # Run with torch.
      # print("run torch")
      torch.backends.cuda.matmul.allow_tf32
      torch.backends.cudnn.allow_tf32
      torch_result = torch.mm(lhs_torch, rhs_torch)

      if argparse.dump_full_tensor:
        torch.set_printoptions(threshold=10_000)
        print(f"torch_result: {torch_result}")
        print(f"torch - baseline_result_0: {torch_result - baseline_result_0.cuda()}")
        print(f"torch - td_result_0: {torch_result - td_result_0.cuda()}")

      if check_results:
        # print("check results")
        # Cross-impl test: this is tricky as we may have a TF32 vs F32 situation.
        # Compute a proper precision for the problem size, assuming TF32 implementation.
        rtol, atol = config.compute_precision(K, lhs_torch, rhs_torch)
        torch.testing.assert_close(
          td_result_0.cuda(), torch_result, rtol=rtol, atol=atol)
        torch.testing.assert_close(
          baseline_result_0.cuda(), torch_result, rtol=rtol, atol=atol)
        torch.testing.assert_close(
          td_result_0.cuda(), baseline_result_0.cuda(), rtol=rtol, atol=atol)
      
        # We already ran an iteration above.
        if n_iters > 1:
          # print(f"run {n_iters - 1} iterations")
          for iter in range(n_iters - 1):
            # init_fn = ones_initialization_fn
            # init_fn = linspace_initialization_fn
            init_fn = randn_initialization_fn
            lhs_torch, rhs_torch, lhs_iree, rhs_iree = \
              make_fill_matmul_f32_tensors(M, N, K, init_fn)
            rtol, atol = config.compute_precision(K, lhs_torch, rhs_torch)

            # Test against self, this should be bitwise accurate to rule out sync issues.
            td_result_0 = torch.from_numpy(td_fun(lhs_iree, rhs_iree).to_host())
            td_result_1 = torch.from_numpy(td_fun(lhs_iree, rhs_iree).to_host())
            torch.testing.assert_close(
              td_result_0.cuda(), td_result_1.cuda(), rtol=1e-07, atol=1e-07)
          
            # Cross-impl test.
            baseline_result = torch.from_numpy(baseline_fun(lhs_iree, rhs_iree).to_host())
            torch.testing.assert_close(
              td_result_0.cuda(), baseline_result.cuda(), rtol=rtol, atol=atol)

