
mlir_type_to_init_val = {
  "f64": "0.0",
  "f32": "0.0",
  "bf16": "0.0",
  "f16": "0.0",
  # "f8": "0.0",
  "i64": "0",
  "i32": "0",
  "i16": "0",
  "i8": "0",
}

fill_matmul_template = """ 
!input_tensor_t = tensor<${M}x${K}x${LHS_TYPE}>
!weight_tensor_t = tensor<${K}x${N}x${RHS_TYPE}>
!output_tensor_t = tensor<${M}x${N}x${RES_TYPE}>
func.func @${FN_NAME}(%in: !input_tensor_t, %wei: !weight_tensor_t) -> !output_tensor_t {
  %cst_0 = arith.constant ${INIT_VAL} : ${RES_TYPE}
  %empty = tensor.empty() : !output_tensor_t
  %out = linalg.fill ins(%cst_0 : ${RES_TYPE}) outs(%empty : !output_tensor_t) -> !output_tensor_t
  %res = linalg.matmul
     ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
    outs(%out: !output_tensor_t) -> !output_tensor_t
  return %res : !output_tensor_t
}
"""

fill_matmul_transpose_a_template = """ 
!input_tensor_t = tensor<${K}x${M}x${LHS_TYPE}>
!weight_tensor_t = tensor<${K}x${N}x${RHS_TYPE}>
!output_tensor_t = tensor<${M}x${N}x${RES_TYPE}>
func.func @${FN_NAME}(%in: !input_tensor_t, %wei: !weight_tensor_t) -> !output_tensor_t {
  %cst_0 = arith.constant ${INIT_VAL} : ${RES_TYPE}
  %empty = tensor.empty() : !output_tensor_t
  %out = linalg.fill ins(%cst_0 : ${RES_TYPE}) outs(%empty : !output_tensor_t) -> !output_tensor_t
  %res = linalg.matmul_transpose_a
     ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
    outs(%out: !output_tensor_t) -> !output_tensor_t
  return %res : !output_tensor_t
}
"""

fill_matmul_transpose_b_template = """ 
!input_tensor_t = tensor<${M}x${K}x${LHS_TYPE}>
!weight_tensor_t = tensor<${N}x${K}x${RHS_TYPE}>
!output_tensor_t = tensor<${M}x${N}x${RES_TYPE}>
func.func @${FN_NAME}(%in: !input_tensor_t, %wei: !weight_tensor_t) -> !output_tensor_t {
  %cst_0 = arith.constant ${INIT_VAL} : ${RES_TYPE}
  %empty = tensor.empty() : !output_tensor_t
  %out = linalg.fill ins(%cst_0 : ${RES_TYPE}) outs(%empty : !output_tensor_t) -> !output_tensor_t
  %res = linalg.matmul_transpose_b
     ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
    outs(%out: !output_tensor_t) -> !output_tensor_t
  return %res : !output_tensor_t
}
"""

supported_matmul_templates = [
  fill_matmul_template, 
  fill_matmul_transpose_a_template,
  fill_matmul_transpose_b_template,
]

def make_fill_matmul_problem(template_str, M, N, K, LHS_TYPE, RHS_TYPE, RES_TYPE, td_config=None):
  if not template_str in supported_matmul_templates:
    raise ValueError(f"Unsupported matmul template: {template_str}")

  fn_name = f"mm_{M}_{N}_{K}"
  fn_name = fn_name if td_config is None else \
    fn_name  + "_" + "_".join([f"{k}_{v}" for k, v in td_config.items()])
  fn_name = fn_name.replace(',', '_')
  return template_str.replace(
    "${M}", str(M)).replace(
    "${K}", str(K)).replace(
    "${N}", str(N)).replace(
    "${LHS_TYPE}", str(LHS_TYPE)).replace(
    "${RHS_TYPE}", str(RHS_TYPE)).replace(
    "${RES_TYPE}", str(RES_TYPE)).replace(
    "${INIT_VAL}", str(mlir_type_to_init_val[RES_TYPE])).replace(
    "${FN_NAME}", str(fn_name)), \
    fn_name

# Some extra flags that may be useful to uncomment, but not to remember and type...
# "--mlir-print-ir-after-all",
# "--iree-hal-dump-executable-benchmarks-to=/tmp/iree-executables",
# Dump generated binary files (i.e. PTX).
# "--iree-hal-dump-executable-binaries-to=/tmp/iree-executables",
# Uncomment the following to see generated bitcode files, on which llvm-dis
# can be used to get to LLVM IR.
# "--iree-hal-dump-executable-intermediates-to=/tmp/iree-executables",
# "--iree-hal-dump-executable-sources-to=/tmp/iree-executables",
def append_td_repro_options(options, td_repro=False):
  return options + [
    "--debug-only=transform-dialect-save-repro",
    "--mlir-disable-threading",
    # IREE dumps.
    "--iree-hal-dump-executable-benchmarks-to=/tmp/iree-executables",
    "--iree-hal-dump-executable-binaries-to=/tmp/iree-executables",
    "--iree-hal-dump-executable-intermediates-to=/tmp/iree-executables",
    "--iree-hal-dump-executable-sources-to=/tmp/iree-executables",
  ] if td_repro else options

def make_iree_baseline_options(td_repro=False):
  res = [
    # Despite best efforts, can't seem to start IREE from GPU-resident tensors 
    # without copies accounted in the cost atm...
    "--iree-hal-benchmark-dispatch-repeat-count=2",
    "--iree-stream-resource-index-bits=64",
    "--iree-vm-target-index-bits=64",
    "--iree-hal-cuda-llvm-target-arch=sm_80",
    "--iree-codegen-llvmgpu-enable-transform-dialect-jit=false",
  ]
  return append_td_repro_options(res, td_repro)


# Some extra flags that may be useful to uncomment, but not to remember and type...
# Uncomment some options manually
# 
# Options for debugging mapping to mma operations.
# "--debug-only=vector-unroll",
# "--debug-only=iree-codegen-gpu-pipelining",
# "--debug-only=llvm-gpu-utils",
#
# Options for debugging GPU barrier removal.
# "--debug-only=transform-llvmgpu-extensions-alias",
#
# Options for debugging transform strategies.
# "--debug-only=iree-transform-builder",
# "--debug-only=transform-dialect-save-repro",
# "--mlir-disable-threading",
def make_iree_td_options(config, td_repro=False, benchmark=False):
  res = [
    # Despite best efforts, can't seem to start IREE from GPU-resident tensors 
    # without copies accounted in the cost atm...
    "--iree-hal-benchmark-dispatch-repeat-count=2",
    "--iree-stream-resource-index-bits=64",
    "--iree-vm-target-index-bits=64",
    "--iree-hal-cuda-llvm-target-arch=sm_80",
  ]

  res = res + [
    f"--iree-codegen-llvmgpu-enable-transform-dialect-aligned-matmul",
    f"--iree-codegen-llvmgpu-enable-transform-dialect-pad-strategy",
    f"--iree-codegen-llvmgpu-enable-transform-dialect-small-matmul",
    f"--iree-flow-enable-pad-handling",
    # Some manual debugging
    # f"--debug-only=iree-gpu-copy-mapping",
    # f"--debug-only=iree-transform-builder",
    # f"--debug-only=mlir-print-ir-after-all",
    # f"--debug-only=mlir-disable-threading",
  ]
  if 'default' in config:
    return append_td_repro_options(res, td_repro)

  res = res + [
    f"--td-matmul-strategy-blk-sizes={config['blk']}",
    f"--td-matmul-strategy-num-threads={config['tds']}",
    f"--td-matmul-strategy-pipeline-depth={config['p']}",
    f"--td-matmul-strategy-reduc-size={config['r']}",
    f"--td-matmul-strategy-use-async-copies={config['acp']}",
  ]
  if 'wps' in config:
    res.append(f"--td-matmul-strategy-num-warps={config['wps']}")
  if 'mma' in config:
    res.append(f"--td-matmul-strategy-use-mma-sync={config['mma']}")
  if 'wmma' in config:
    res.append(f"--td-matmul-strategy-use-wmma={config['wmma']}")
  if 'fma' in config:
    res.append(f"--td-matmul-strategy-use-fma={config['fma']}")
  if 'peel' in config:
    res.append(f"--td-matmul-strategy-peel-pipeline-epilogue={config['peel']}")
  return append_td_repro_options(res, td_repro)

def append_td_graph_script(l, filename=None):
  return l + [
    # TODO: when openxla builds python bindings properly we can just add
    # f'--openxla-transform-preprocessing={filename}',
    # and later just omit it altogether because the strategy is embedded into 
    # the C++ plugin.
    # In the meantime we need to use IREE.
    f'--iree-flow-dispatch-use-transform-dialect={filename}',
  ] if filename is not None else l

# For now assume type if TF32.
def compute_precision(K, LHS_TYPE, RHS_TYPE, RES_TYPE, *tensors):
  max_value = 0.0
  for t in tensors:
      max_value = max(float(t.abs().max()), max_value)
  # Relative precision for TF32 is 1e-4, for FP32 it is 1e-7.
  rtol = 1e-4
  if LHS_TYPE == "i8" or RHS_TYPE == "i8" or RES_TYPE == "i8":
    rtol = 1e-1
  elif LHS_TYPE == "bf16" or RHS_TYPE == "bf16" or RES_TYPE == "bf16":
    rtol = 1e-2
  elif LHS_TYPE == "f16" or RHS_TYPE == "f16" or RES_TYPE == "f16":
    rtol = 1e-3
  atol = rtol * max_value * K
  # print(f"rtol={rtol} atol={atol}")
  return rtol, atol
