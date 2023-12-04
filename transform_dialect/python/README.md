# Python Verification Scripts

This directory contains a few simple python scripts using IREE python bindings
to aid in correctness verification of new codegen strategies.

## Setup

It is recommended to use a python virtual environment which can be setup with
```
python -m venv iree_sample_venv
```
Or alternatively use the one that was setup when building the IREE runtime. The
only external dependency is PyTorch, which is only used for it's testing library
to do the numerical comparison of values. This can be installed from this directory with
```
pip install -r requirements.txt
```

To run the scripts IREE compiler and runtime python bindings will need to have been built.
This can be done my making sure to set the appropriate CMake flags.
```
   -DIREE_BUILD_PYTHON_BINDINGS=ON
   -DPython3_EXECUTABLE="$(which python)"
```

From here, navigate to the the build directory and export the python path.
```
source .env && export PYTHONPATH
```

These steps can also be found on the IREE website at
https://openxla.github.io/iree/building-from-source/getting-started/#using-the-python-bindings

## Usage

An example usage of the comparator script is as follows
```
python compile_and_compare.py --module matmul.mlir --function=matmul --input_types="32x32xf16 32x16xf16" --device=vulkan --target_info_flag="--iree-vulkan-target-triple=rdna3-unknown-linux" --argset1="--iree-spirv-enable-transform-dialect-jit=true"
```
where `matmul.mlir` resides in this directory and contains
```mlir
!input_tensor_t = tensor<32x32xf16>
!weight_tensor_t = tensor<32x16xf16>
!output_tensor_t = tensor<32x16xf16>
func.func @matmul(%in: !input_tensor_t, %wei: !weight_tensor_t) -> !output_tensor_t {
  %cst_0 = arith.constant 0.0 : f16
  %empty = tensor.empty() : !output_tensor_t
  %out = linalg.fill ins(%cst_0 : f16) outs(%empty : !output_tensor_t) -> !output_tensor_t
  %res = linalg.matmul
     ins(%in, %wei: !input_tensor_t, !weight_tensor_t)
    outs(%out: !output_tensor_t) -> !output_tensor_t
  return %res : !output_tensor_t
}
```

The details for individual flags can be printed with `--help`. A note about
`--argset1`, to specify the compiler flags there has to be an `=` between the
whitespace separated string of flags and the `--argset1` flag, else the
argument parser will fail to parse the flags.

There is another mode available for comparing already compiled modules by using
the `--vmfb1_path` and `--vmfb2_path` flags. Currently the only supported data types
are f32 and f16 due to the way the inputs are randomly generated. This can be improved in the future.
