# Prototype of next generation IREE->JAX API

IREE's current Jax API (`iree.jax`) is based around single function export
of the original XLA protobuf. It has a number of limitations:

* It can only compile one function at a time
* It is not possible to manipulate state
* It is not possible to emit more complicated procedural logic for training
  or inference (in the presence of state)

This prototype uses the new direct MLIR export path in Jax to attempt a more
comprehensive interface that allows the compilation of a complete model,
including capturing globals, initialization, update, and auxiliary functions.

Currently, the user-level API is not yet in place, so using it is more
intrusive than ultimately imagined (i.e. it should be possible to be
somewhat similar in structure to a Torch Module or TF Module as a unit of
exporting a program).

Some IREE and Jax patches are needed until things stabilize. Running the
Jax side with `JAX_ENABLE_MLIR=1` set is required until a better way to opt
in to such behavior is available.

## Saving the Mnist model:

See `examples/high_level_mnist_export.py`:

```
python examples/high_level_mnist_export.py /tmp/mnist_export
```

## Offline training and evaluation:

See `examples/run_trainer.py` for a self contained Python program, using only
IREE's runtime API which will initialize the model, manage checkpoints and
run training steps:

```
python examples/run_trainer.py /tmp/mnist_export/mnist_train.vmfb /tmp/mnist_export/checkpoint.npz
```

More complicated configurations are possible (compiling to run on devices,
embedding with the C or Java APIs, etc).
