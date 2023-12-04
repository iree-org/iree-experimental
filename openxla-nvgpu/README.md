# OpenXLA NVIDIA GPU Compiler and Runtime

This project contains the compiler and runtime plugins enabling specialized
targeting of the OpenXLA platform the NVIDIA GPUs. It builds on top of the
core IREE toolkit.

## Development setup

The project can be built either as part of IREE by manually specifying
plugin paths via `-DIREE_COMPILER_PLUGIN_PATHS`, or for development tailored
to NVIDIA GPUs specifically, can be built directly:

```
cmake -GNinja -B build/ -S . \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DIREE_ENABLE_ASSERTIONS=ON \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DIREE_ENABLE_LLD=ON

# Recommended:
# -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
```
