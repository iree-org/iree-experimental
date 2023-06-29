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

zeros_initialization_fn = lambda x, y: torch.zeros(x, y)
ones_initialization_fn = lambda x, y: torch.ones(x, y)
linspace_initialization_fn = lambda x, y: torch.linspace(0, x*y - 1, x*y).reshape(x, y)
randn_initialization_fn = lambda x, y: torch.randn(x, y)

mlir_type_to_dtype = {
  "f64": torch.float64,
  "f32": torch.float32,
  "bf16": torch.bfloat16,
  "f16": torch.float16,
  # "f8": torch.float8,
  "i64": torch.int64,
  "i32": torch.int32,
  "i16": torch.int16,
  "i8": torch.int8,
}

def convert(t, type):
  return t.to(mlir_type_to_dtype[type])

# General Qs to avoid working around the system:
#   * why does IREE have asymmetric APIs such as tensor.to_host() but ireert.asdevicearray(tensor) ?
#   * why not have something to build an IREE device tensor from a torch cuda tensor ?
#     - do we really need all these genuflexions through the host and maintaining 3 array copies ?
#   * either I am misusing this API or ireert.asdevicearray does not work as expected:
#     - the kernel cost as shown in nvprof is still heavily skewed by the copy cost
#     - it is still necessary to run multiple times to show the real timings
def finalize_make_fill_matmul_tensors(lhs, rhs, res, LHS_TYPE, RHS_TYPE, RES_TYPE):
  lhs = convert(lhs, LHS_TYPE)
  rhs = convert(rhs, RHS_TYPE)
  res = convert(res, RES_TYPE)
  return lhs.cuda(), \
         rhs.cuda(), \
         res.cuda(), \
         ireert.asdevicearray(iree_runtime_device, lhs, implicit_host_transfer=False), \
         ireert.asdevicearray(iree_runtime_device, rhs, implicit_host_transfer=False),  \
         ireert.asdevicearray(iree_runtime_device, res, implicit_host_transfer=False)

def make_fill_matmul_tensors(M, N, K, LHS_TYPE, RHS_TYPE, RES_TYPE, initialization_fn=lambda x, y: torch.randn(x, y)):
  lhs, rhs, res = initialization_fn(M, K), initialization_fn(K, N), zeros_initialization_fn(M, N)
  return finalize_make_fill_matmul_tensors(lhs, rhs, res, LHS_TYPE, RHS_TYPE, RES_TYPE)

def torch_baseline_fill_matmul_tensors(lhs, rhs, out=None):
  return torch.mm(lhs, rhs, out=out)


def make_fill_matmul_transpose_a_tensors(M, N, K, LHS_TYPE, RHS_TYPE, RES_TYPE, initialization_fn=lambda x, y: torch.randn(x, y)):
  lhs, rhs, res = initialization_fn(K, M), initialization_fn(K, N), zeros_initialization_fn(M, N)
  return finalize_make_fill_matmul_tensors(lhs, rhs, res, LHS_TYPE, RHS_TYPE, RES_TYPE)

def torch_baseline_fill_matmul_transpose_a_tensors(lhs, rhs, out=None):
  return torch.mm(lhs.t(), rhs, out=out)


def make_fill_matmul_transpose_b_tensors(M, N, K, LHS_TYPE, RHS_TYPE, RES_TYPE, initialization_fn=lambda x, y: torch.randn(x, y)):
  lhs, rhs, res = initialization_fn(M, K), initialization_fn(N, K), zeros_initialization_fn(M, N)
  return finalize_make_fill_matmul_tensors(lhs, rhs, res, LHS_TYPE, RHS_TYPE, RES_TYPE)

def torch_baseline_fill_matmul_transpose_b_tensors(lhs, rhs, out=None):
  return torch.mm(lhs, rhs.t(), out=out)



def run(problem_sizes,
        data_types,
        td_configurations,
        argparse,
        n_iters,
        template_str = config.fill_matmul_template,
        tensor_builder_fn = make_fill_matmul_tensors,
        torch_baseline_fn = torch_baseline_fill_matmul_tensors,
        check_results=False):
  for M, N, K in problem_sizes:
    for LHS_TYPE, RHS_TYPE, RES_TYPE in data_types:
      # init_fn = ones_initialization_fn
      # init_fn = linspace_initialization_fn
      init_fn = randn_initialization_fn
      ir_str, fn_name = config.make_fill_matmul_problem(
        template_str, M, N, K, LHS_TYPE, RHS_TYPE, RES_TYPE)
      print(f"matmul_{M}x{N}x{K}")
      for td_config in td_configurations:
        print(f"td_config: {td_config}")
        td_ir_str, td_fn_name = config.make_fill_matmul_problem(
          template_str, M, N, K, LHS_TYPE, RHS_TYPE, RES_TYPE, td_config)
        lhs_torch, rhs_torch, res_torch, lhs_iree, rhs_iree, res_iree = \
          tensor_builder_fn(M, N, K, LHS_TYPE, RHS_TYPE, RES_TYPE, init_fn)

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

        if check_results:
          # print("check results")
          # Cross-impl test: this is tricky as we may have a TF32 vs F32 situation.
          # Compute a proper precision for the problem size, assuming TF32 implementation.
          rtol, atol = config.compute_precision(K, LHS_TYPE, RHS_TYPE, RES_TYPE, lhs_torch, rhs_torch)

          # Run with torch, only if types match for now, don't know how to do
          # mixed precision in torch.
          # print("run torch")
          torch_result = None
          if LHS_TYPE == RHS_TYPE and LHS_TYPE == RES_TYPE:
            torch.backends.cuda.matmul.allow_tf32
            torch.backends.cudnn.allow_tf32
            torch_result = torch_baseline_fn(lhs_torch, rhs_torch, out=res_torch)
            if argparse.dump_full_tensor:
              torch.set_printoptions(threshold=10_000)
              print(f"torch_result: {torch_result}")
              print(f"torch - td_result_0: {torch_result - td_result_0.cuda()}")

          if torch_result is not None:
            torch.testing.assert_close(
              td_result_0.cuda(), torch_result, rtol=rtol, atol=atol)
          
          for iter in range(n_iters - 1):
            # init_fn = ones_initialization_fn
            # init_fn = linspace_initialization_fn
            init_fn = randn_initialization_fn
            lhs_torch, rhs_torch, res_torch, lhs_iree, rhs_iree, res_iree = \
              tensor_builder_fn(M, N, K, LHS_TYPE, RHS_TYPE, RES_TYPE, init_fn)
            rtol, atol = config.compute_precision(K, LHS_TYPE, RHS_TYPE, RES_TYPE, lhs_torch, rhs_torch)

            # Test against self, this should be bitwise accurate to rule out sync issues.
            td_result_0 = torch.from_numpy(td_fun(lhs_iree, rhs_iree).to_host())
            td_result_1 = torch.from_numpy(td_fun(lhs_iree, rhs_iree).to_host())
            torch.testing.assert_close(
              td_result_0.cuda(), td_result_1.cuda(), rtol=1e-07, atol=1e-07)
          
            # Cross-impl test.
            # Compile and run the baseline, only if LHS and RHS types match for now.
            if torch_result is not None:
              torch_result = torch_baseline_fn(lhs_torch, rhs_torch, out=res_torch)
              torch.testing.assert_close(
                td_result_0.cuda(), torch_result, rtol=rtol, atol=atol)

