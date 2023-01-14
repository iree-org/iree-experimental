# IREE PJRT Plugin

This directory contains an experimental PJRT plugin library which can bridge
Jax (and TensorFlow in the future) to IREE. This will eventually become
a separate repository.

# Developing

Support for dynamically loaded PJRT plugins is brand new as of 12/21/2022 and
there are sharp edges still. The following procedure is being used to develop.

It is recommended to checkout `jax`, `iree`, `iree-samples`, and `tensorflow`
side by side, as we will be overriding them all to build at head.

Note that although Jax is emitting serialized stablehlo bytecode, which should
be compatible across versions eventually, it is still early days and things
are not stable yet. It is recommended to use a `tensorflow` repo at the same
commit that IREE uses.

## Build and install custom jaxlib

From a Jax checkout:

```
# Currently pluggable PJRT is commingled with TPU support... folks are
# working on it :/
pip install -e .
python build/build.py \
  --bazel_options=--override_repository=org_tensorflow=$PWD/../tensorflow \
  --enable_tpu
pip install dist/*.whl --force-reinstall
```

## Build this project and look at a plugin

```
bazel build ...
IREE_PLUGIN_PATH="$PWD/bazel-bin/iree/integrations/pjrt/cpu/lib_pjrt_plugin_iree_cpu.so"
ls -lh $IREE_PLUGIN_PATH
```

## Run a Jax test program.

Note that the JAX plugin initialization sequence needs a patch:
https://github.com/google/jax/pull/14011

```
# Tells the IREE plugin where to find the compiler. Only needed for now.
export IREE_PJRT_COMPILER_LIB_PATH=$IREE_BUILD_DIR/lib/libIREECompiler.so
export PJRT_NAMES_AND_LIBRARY_PATHS="iree_cpu:$PWD/bazel-bin/iree/integrations/pjrt/cpu/pjrt_plugin_iree_cpu.so,iree_cuda:$PWD/bazel-bin/iree/integrations/pjrt/cuda/pjrt_plugin_iree_cuda.so"
# Jax only enable the plugin path if TPUs enabled for the moment.
export JAX_USE_PJRT_C_API_ON_TPU=1

# Optional: path to libcuda.so
# export LD_LIBRARY_PATH=/usr/lib/wsl/lib

JAX_PLATFORMS=iree_cpu python test/test_simple.py
JAX_PLATFORMS=iree_cuda python test/test_simple.py
```

## Generating runtime traces

The plugins can be build with tracing enabled by adding the bazel build flag
`--iree_enable_runtime_tracing`. With this flag, if a profiler is running,
instrumentation will be sent to it. It can be useful to set the environment
variable `TRACY_NO_EXIT=1` in order to block termination of one-shot programs
that exit too quickly to stream all events.
