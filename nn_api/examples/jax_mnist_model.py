# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import time
import itertools

import numpy.random as npr

import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax
from examples import datasets


def loss(params, batch):
  inputs, targets = batch
  preds = predict(params, inputs)
  return -jnp.mean(jnp.sum(preds * targets, axis=1))

def accuracy(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return jnp.mean(predicted_class == target_class)

init_random_params, predict = stax.serial(
    Dense(1024), Relu,
    Dense(1024), Relu,
    Dense(10), LogSoftmax)

if __name__ == "__main__":
  rng = random.PRNGKey(0)

  step_size = 0.001
  num_epochs = 10
  batch_size = 128
  momentum_mass = 0.9

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

  opt_init, opt_update, get_params = optimizers.momentum(step_size, mass=momentum_mass)

  @jit
  def update(i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  @jit
  def initialize(rng):
    _, init_params = init_random_params(rng, (-1, 28 * 28))
    return init_params

  print("INIT_RANDOM_PARAMS:::")
  mhlo_text = initialize.lower(rng)._xla_computation()
  import iree.compiler.ir
  import iree.compiler.passmanager
  import iree.compiler.api.driver
  from iree.compiler.dialects import (
    mhlo as mhlo_d,
    chlo as chlo_d,
  )
  #from iree.compiler.tools import xla as xlac
  with iree.compiler.ir.Context() as context:
    mhlo_d.register_mhlo_dialect(context)
    chlo_d.register_chlo_dialect(context)
    module = iree.compiler.ir.Module.parse(mhlo_text)
    pm = iree.compiler.passmanager.PassManager()
    iree.compiler.api.driver.build_xla_cleanup_pass_pipeline(pm)
    iree.compiler.api.driver.build_mhlo_import_pass_pipeline(pm)
    pm.run(module)
    print(module)
  # iree_text = xlac.compile_str(mhlo_text, import_only=True,
  #   import_format="mlir_text")

  # import_extra_args=[
  #   "--xla-format=mlir_text",
  # ])
  #print(str(iree_text))
  #_, init_params = init_random_params(rng, (-1, 28 * 28))
  # opt_state = opt_init(init_params)
  # itercount = itertools.count()

  # print("\nStarting training...")
  # for epoch in range(num_epochs):
  #   start_time = time.time()
  #   for _ in range(num_batches):
  #     opt_state = update(next(itercount), opt_state, next(batches))
  #   epoch_time = time.time() - start_time

  #   params = get_params(opt_state)
  #   train_acc = accuracy(params, (train_images, train_labels))
  #   test_acc = accuracy(params, (test_images, test_labels))
  #   print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
  #   print("Training set accuracy {}".format(train_acc))
  #   print("Test set accuracy {}".format(test_acc))
