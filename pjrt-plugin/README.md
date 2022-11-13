# IREE PJRT Plugin

This directory contains an experimental PJRT plugin library which can bridge
Jax (and TensorFlow in the future) to IREE. This will eventually become
a separate repository.

## Design Notes

This implementation uses builds a shared library which exports the PJRT
C API. See the following in the TensorFlow/XLA repo for how this works:

* `pjrt_c_api_client.cc` : `GetCApiClient()` is hard-coded to search for
  a TPU shared library. We patch this resolution procedure to enable more of
  a real plugin discovery phase. This needs more work obviously.
* See `pjrt_c_api.h` for the definition of the C API. Note that this only
  defines the structure of functions, not the discovery mechanism.
* The `libIreePjrtPlugin.so` library hard-links to the IREE runtime and 
  dynamically loads the compiler. This allows for a build/optionality point 
  which makes the plugin lightweight and allows the compiler to be shared in 
  various ways (and enables it to be built/installed separately).
