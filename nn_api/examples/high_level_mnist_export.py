# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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

from iree.jax2 import exporter
from iree.jax2.builtins import *


def main(args):
  exp = exporter.ExportModule.create_empty()
  build_model(exp)

  print("Saving...")
  output_dir = args[0]
  # TODO: Switch to binary=True once fixed: If module doesn't verify, it tries
  # to output text.
  with open(os.path.join(output_dir, "mnist_train.mlir"), "w") as f:
    exp.module.operation.print(f, binary=False)

  print("Compiling...")
  compile_out_of_process(exp.module,
                         os.path.join(output_dir, "mnist_train.vmfb"),
                         input_type="mhlo")


def build_model(exp: exporter.ExportModule):
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

  # NOTE: The export_pure_func annotation is unfortunate and we should be able
  # to eliminate it. What it says is that this function should be jitted when
  # exporting. Probably just using @jax.jit for this would be best, but I
  # couldn't find a good way to detect whether that target of a call was the
  # result of @jax.jit and that would be required.
  @export_pure_func
  def predict_target_class(params, inputs):
    # TODO: An issue with argmax (https://github.com/google/iree/issues/7748).
    #predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    #return predicted_class
    prediction = predict(params, inputs)
    return prediction

  rng = random.PRNGKey(0)
  _, init_params = init_random_params(rng, (-1, 28 * 28))
  opt_init, opt_update, opt_get_params = optimizers.momentum(0.001, mass=0.9)
  opt_state = opt_init(init_params)

  exp_params = exp.def_global_tree("init_params", init_params)
  exp_opt_state = exp.def_global_tree("opt_state", opt_state)

  example_batch = get_example_batch()

  @export_pure_func
  def initialize_optimizer(rng):
    _, init_params = init_random_params(rng, (-1, 28 * 28))
    return opt_init(init_params)

  @export_pure_func
  def update_step(batch, opt_state):
    params = opt_get_params(opt_state)
    # TODO: It appears that since the iteration count isn't used in this
    # computation, it gets elided from the function signature.
    # Just setting the first arg to None for this demo.
    # It seems likely that we want to store the iteration count as a global
    # anyway and tie it.
    # Note that this may be a bug in the MLIR lowerings: the XLA lowering
    # does some special things to preserve dead arguments.
    return opt_update(None, grad(loss)(params, batch), opt_state)

  @exp.def_func
  def get_params():
    return exp_params

  @exp.def_func
  def get_opt_state():
    return exp_opt_state

  @exp.def_func(arguments=[exp_opt_state])
  def set_opt_state(opt_state):
    store_global(exp_opt_state, opt_state)

  @exp.def_func(arguments=[rng])
  def initialize(rng):
    store_global(exp_opt_state, initialize_optimizer(rng))

  @exp.def_func(arguments=[example_batch])
  def update(batch):
    new_opt_state = update_step(batch, exp_opt_state)
    store_global(exp_opt_state, new_opt_state)

  @exp.def_func(arguments=[example_batch[0]])
  def predict(inputs):
    return predict_target_class(exp_params, inputs)


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


def compile_out_of_process(root_module, output_path, input_type):
  iree_tools.compile_str(str(root_module),
                         target_backends=["cpu"],
                         output_file=output_path,
                         input_type=input_type)


main(sys.argv[1:])
