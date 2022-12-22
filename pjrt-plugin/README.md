# IREE PJRT Plugin

This directory contains an experimental PJRT plugin library which can bridge
Jax (and TensorFlow in the future) to IREE. This will eventually become
a separate repository.

# Developing

Support for dynamically loaded PJRT plugins is brand new as of 12/21/2022 and
there are sharp edges still. The following procedure is being used to develop.

It is recommended to checkout `jax`, `iree`, `iree-samples`, and `tensorflow`
side by side, as we will be overriding them all to build at head.

## Build and install custom jaxlib

From a Jax checkout:

```
# Currently pluggable PJRT is commingled with TPU support... folks are
# working on it :/
pip install -e .
python build/build.py \
  --bazel_options=--override_repository=org_tensorflow=../tensorflow \
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

```
# Tells the IREE plugin where to find the compiler. Only needed for now.
export IREE_PJRT_COMPILER_LIB_PATH=$IREE_BUILD_DIR/lib/libIREECompiler.so
export PJRT_NAMES_AND_LIBRARY_PATHS="iree_cpu:$IREE_PLUGIN_PATH"
# Jax only enable the plugin path if TPUs enabled for the moment.
export JAX_USE_PJRT_C_API_ON_TPU=1

python test_init.py
```

### test_init.py

```
import jax
from jax._src.lib import xla_bridge
from jax._src.lib import xla_client

def iree_backend_factory():
  # Can also load in code:
  # xla_client._xla.load_pjrt_plugin("iree_cpu", "/path/to/plugin.so")
  return xla_client._xla.get_c_api_client("iree_cpu")

xla_bridge.register_backend_factory("iree_cpu", iree_backend_factory)

# actually this is a better way to make sure jax uses your plugin, it'll raise an error if it can't initialize
jax.config.update("jax_platforms", "iree_cpu")


print(jax.numpy.add(1, 1))
```
