# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Usage: python high_level_mnist_export.py {dest dir}

import os
import sys

import numpy.random as npr
from examples import datasets

from iree.compiler import (
    tools as iree_tools,)

import jax.core
import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax
from jax.interpreters.xla import abstractify

from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           register_pytree_node)

from iree.jax2.staging_api import *
from iree.jax2.builtins import *


def main(args):
  output_dir = args[0]
  os.makedirs(output_dir, exist_ok=True)
  jax.config.update("jax_enable_mlir", True)
  staged_module = build_model()

  print("Saving mlir...")
  with open(os.path.join(output_dir, "mnist_train.mlir"), "wb") as f:
    get_mlir_module(staged_module).operation.print(f, binary=True)

  print("Compiling...")
  compiled_module = staged_module()

  print("Saving binary...")
  with open(os.path.join(output_dir, "mnist_train.vmfb"), "wb") as f:
    f.write(get_compiled_binary(compiled_module))


def build_model():
  init_random_params, predict = stax.serial(
      Dense(1024),
      Relu,
      Dense(1024),
      Relu,
      Dense(10),
      LogSoftmax,
  )

  def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))

  rng = random.PRNGKey(0)
  _, init_params = init_random_params(rng, (-1, 28 * 28))
  opt_init, opt_update, opt_get_params = optimizers.momentum(0.001, mass=0.9)
  opt_state = opt_init(init_params)

  example_batch = get_example_batch()

  # Putting together the class which extends StagedModule implicitly assembles
  # the corresponding MLIR module.
  class MnistModule(StagedModule):
    _params = export_global(init_params)
    _opt_state = export_global(opt_state)

    @export_kernel
    def _initialize_optimizer(_, rng):
      _, init_params = init_random_params(rng, (-1, 28 * 28))
      return opt_init(init_params)

    @export_kernel
    def _update_step(_, batch, opt_state):
      params = opt_get_params(opt_state)
      # TODO: It appears that since the iteration count isn't used in this
      # computation, it gets elided from the function signature.
      # Just setting the first arg to None for this demo.
      # It seems likely that we want to store the iteration count as a global
      # anyway and tie it.
      # Note that this may be a bug in the MLIR lowerings: the XLA lowering
      # does some special things to preserve dead arguments.
      return opt_update(None, grad(loss)(params, batch), opt_state)

    @export_kernel
    def _predict_target_class(mdl, params, inputs):
      # TODO: An issue with argmax (https://github.com/google/iree/issues/7748).
      #predicted_class = jnp.argmax(predict(params, inputs), axis=1)
      #return predicted_class
      prediction = predict(params, inputs)
      return prediction

    @export_traced_proc
    def get_params(mdl):
      return mdl._params

    @export_traced_proc
    def get_opt_state(mdl):
      return mdl._opt_state

    @export_traced_proc(signature=[opt_state])
    def set_opt_state(mdl, new_opt_state):
      store_global(mdl._opt_state, new_opt_state)

    @export_traced_proc(signature=[rng])
    def initialize(mdl, rng):
      store_global(mdl._opt_state, mdl._initialize_optimizer(rng))

    @export_traced_proc(signature=[example_batch])
    def update(mdl, batch):
      new_opt_state = mdl._update_step(batch, mdl._opt_state)
      store_global(mdl._opt_state, new_opt_state)

    @export_traced_proc(signature=[example_batch[0]])
    def predict(mdl, inputs):
      return mdl._predict_target_class(mdl._params, inputs)

  return MnistModule


def get_example_batch():
  batch_size = 128
  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  num_batches = num_complete_batches + bool(leftover)

  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]

  batches = data_stream()
  return next(batches)


main(sys.argv[1:])
