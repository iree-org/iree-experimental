# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import os
import sys

from iree.compiler import (
  ir,
  passmanager,
)
from iree.compiler.api import driver as iree_driver
from iree.nn.jax_utils import (
    JaxImportContext,)
from iree.nn.module_builder import (
    ModuleBuilder,)

import numpy.random as npr
import jax.core
import jax.numpy as jnp
from jax import jit, grad, random
from jax.example_libraries import optimizers
from jax.example_libraries import stax
from jax.example_libraries.stax import Dense, Relu, LogSoftmax
from examples import datasets

from jax.tree_util import (tree_map, tree_flatten, tree_unflatten,
                           register_pytree_node)


def main(args):
  output_path = args[0]
  example_batch = get_example_batch()
  imp = import_mnist_model(example_batch)
  with open(os.path.join(output_path, "iree_input.mlir"), "wb") as f:
    imp.ic.module.print(f, binary=True)

  # Compiler it.
  # TODO: Make it possible to use Operations with pass manager API.
  root_module = imp.ic._root_module
  compiler_options = iree_driver.CompilerOptions()
  # TODO: CPU broken:
  # error: 'linalg.matmul' op expected op to be distributed along 2 dimensions
  # compiler_options.add_target_backend("cpu")
  # TODO: Vulkan broken:
  # python: /home/stella/src/iree/iree/compiler/Codegen/SPIRV/KernelConfig.cpp:365: mlir::LogicalResult mlir::iree_compiler::setDefaultOpConfig(spirv::ResourceLimitsAttr, mlir::Operation *): Assertion `partitionedLoops.size() == tiledLoopInfo.size()' failed.
  # compiler_options.add_target_backend("vulkan")
  # TODO: CUDA broken:
  # python: /home/stella/src/iree/third_party/llvm-project/llvm/include/llvm/ADT/SmallVector.h:277: llvm::SmallVectorTemplateCommon::reference llvm::SmallVectorTemplateCommon<llvm::StringMap<mlir::OpPassManager>>::operator[](llvm::SmallVectorTemplateCommon::size_type) [T = llvm::StringMap<mlir::OpPassManager>]: Assertion `idx < size()' failed.
  # compiler_options.add_target_backend("cuda")
  with imp.ic.context:
    pm = passmanager.PassManager()
    iree_driver.build_iree_vm_pass_pipeline(compiler_options, pm)
    pm.run(root_module)
  with open(os.path.join(output_path, "mnsit_train.vmfb"), "wb") as f:
    iree_driver.translate_module_to_vm_bytecode(compiler_options,
                                                root_module, f)


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


def import_mnist_model(example_batch):
  imp = ModuleBuilder(JaxImportContext())
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

  def predict_target_class(params, inputs):
    # TODO: An issue with argmax (https://github.com/google/iree/issues/7748).
    #predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    #return predicted_class
    prediction = predict(params, inputs)
    return prediction

  rng = random.PRNGKey(0)
  _, init_params = init_random_params(rng, (-1, 28 * 28))
  opt_init, opt_update, get_params = optimizers.momentum(0.001, mass=0.9)
  opt_state = opt_init(init_params)

  print(init_params.__class__)
  print(opt_state.__class__)

  example_batch_flat, example_batch_tree_def = tree_flatten(example_batch)
  opt_state_flat, tree_def = tree_flatten(opt_state)
  opt_state_globals = imp.import_globals(opt_state_flat,
                                         "opt_state",
                                         mutable=True,
                                         initialize=False)
  print(f"Opt state tree def: {tree_def}")

  # Import an initialize function.
  @jit
  def jit_initialize(rng):
    _, init_params = init_random_params(rng, (-1, 28 * 28))
    return opt_init(init_params)

  mhlo_text = jit_initialize.lower(rng)._xla_computation()
  initialize_name = imp.import_function(mhlo_text, "main")

  # Import a get_params function.
  # Doesn't work because it is "trivial".
  # @jit
  # def jit_get_params(opt_state):
  #   return get_params(opt_state)
  # mhlo_text = jit_get_params.lower(opt_state)._xla_computation()
  # print(mhlo_text)

  # Import an update function.
  @jit
  def jit_opt_update(batch, opt_state):
    params = get_params(opt_state)
    # TODO: It appears that since the iteration count isn't used in this
    # computation, it gets elided from the function signature. I'm not sure
    # there is anything that can be done at this level of abstraction to
    # normalize that. Just setting the first arg to None for this demo.
    # It seems likely that we want to store the iteration count as a global
    # anyway and tie it.
    return opt_update(None, grad(loss)(params, batch), opt_state)

  mhlo_text = jit_opt_update.lower(example_batch, opt_state)._xla_computation()
  update_name = imp.import_function(mhlo_text, "main")

  fb = imp.create_function(
      "update",
      imp.get_function_type(update_name).inputs[0:len(example_batch_flat)])
  loaded_opt_state = fb.emit_global_group_load(opt_state_globals)
  updated_opt_state = fb.emit_call(update_name, fb.arguments + loaded_opt_state)
  fb.emit_global_group_store(opt_state_globals, updated_opt_state)
  fb.emit_return()

  # Import a predict function.
  # TODO: This is *super* gross. Something in the MLIR lowering path is
  # DCE'ing unused operands to the function (i.e. the predict function only
  # uses a subset of the optimizer weights) but there is not a good way to get
  # this back at this level. Note that the xla_computation route does preserve
  # this in the HLO (explicitly emitting placeholders for the dead arguments),
  # so this may be a bug in the MLIR lowerings. However, it is also a weakness
  # of what we are doing here at this level. I expect that resolving this
  # as part of the JAX infra will yield a better solution.
  def jit_predict(opt_state, inputs):
    params = get_params(opt_state)
    return predict_target_class(params, inputs)

  #xla_comp = jax.xla_computation(jit_predict)(opt_state, example_batch[0])
  #print(dir(xla_comp))
  #print(xla_comp.as_hlo_text())
  mhlo_text = jit(jit_predict).lower(opt_state,
                                     example_batch[0])._xla_computation()
  predict_name = imp.import_function(mhlo_text, "main")
  fb = imp.create_function("predict",
                           [imp.get_function_type(predict_name).inputs[-1]])
  loaded_opt_state = fb.emit_global_group_load(opt_state_globals)
  used_opt_state = [
      loaded_opt_state[0], loaded_opt_state[2], loaded_opt_state[4],
      loaded_opt_state[6], loaded_opt_state[8], loaded_opt_state[10]
  ]
  fb.emit_return(*fb.emit_call(predict_name, used_opt_state + fb.arguments))
  assert imp.ic.module.verify(), "Predict function did not verify"

  # Opt state initializer.
  fb = imp.create_function("initialize",
                           imp.get_function_type(initialize_name).inputs)
  initial_values = fb.emit_call(initialize_name, fb.arguments)
  fb.emit_global_group_store(opt_state_globals, initial_values)
  fb.emit_return()

  # Get opt state.
  fb = imp.create_function("get_opt_state", [])
  fb.emit_return(*fb.emit_global_group_load(opt_state_globals))

  # Set opt state.
  fb = imp.create_function("set_opt_state", opt_state_globals.flattened_types)
  fb.emit_global_group_store(opt_state_globals, fb.arguments)
  fb.emit_return()

  return imp


main(sys.argv[1:])
