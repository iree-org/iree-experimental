
fill_matmul_f32 = """ 
!input_tensor_t = tensor<${M}x${K}xf32>
!weight_tensor_t = tensor<${K}x${N}xf32>
!output_tensor_t = tensor<${M}x${N}xf32>
func.func @${FN_NAME}(%in: !input_tensor_t, %wei: !weight_tensor_t) -> !output_tensor_t {
  %cst_0 = arith.constant 0.0 : f32 
  %empty = tensor.empty() : !output_tensor_t
  %out = linalg.fill ins(%cst_0 : f32) outs(%empty : !output_tensor_t) -> !output_tensor_t
  %res = linalg.matmul
     ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
    outs(%out: !output_tensor_t) -> !output_tensor_t
  return %res : !output_tensor_t
}
"""

def make_fill_matmul_f32_problem(M, N, K, td_config=None):
  fn_name = "matmul" if td_config is None else "matmul" + "_".join([f"{k}_{v}" for k, v in td_config.items()])
  return fill_matmul_f32.replace("${M}", str(M)).replace("${K}", str(K)).replace("${N}", str(N)).replace("${FN_NAME}", str(fn_name)), \
    fn_name

def make_iree_baseline_options():
  return [
    "--iree-stream-resource-index-bits=64",
    "--iree-vm-target-index-bits=64",
    "--iree-hal-cuda-llvm-target-arch=sm_80",
    "--iree-codegen-llvmgpu-enable-transform-dialect-jit=false",
    ]

def make_iree_td_options(config, debug_td=False):
  res = [
    "--iree-stream-resource-index-bits=64",
    "--iree-vm-target-index-bits=64",
    "--iree-hal-cuda-llvm-target-arch=sm_80",
    f"--td-matmul-strategy-blk-size-x={config['bx']}",
    f"--td-matmul-strategy-blk-size-y={config['by']}",
    f"--td-matmul-strategy-blk-size-z={config['bz']}",
    f"--td-matmul-strategy-num-threads-x={config['tx']}",
    f"--td-matmul-strategy-num-threads-y={config['ty']}",
    f"--td-matmul-strategy-num-threads-z={config['tz']}",
    f"--td-matmul-strategy-num-warps-x={config['wx']}",
    f"--td-matmul-strategy-num-warps-y={config['wy']}",
    f"--td-matmul-strategy-num-warps-z={config['wz']}",
    f"--td-matmul-strategy-pipeline-depth={config['pipe_depth']}",
    f"--td-matmul-strategy-reduc-size={config['red_sz']}",
    f"--td-matmul-strategy-use-async-copies={config['async_cp']}",
    f"--td-matmul-strategy-use-mma-sync={config['mma_sync']}",
  ]
  return res + [
    "--debug-only=transform-dialect-save-repro",
    "--mlir-disable-threading",
  ] if debug_td else res
