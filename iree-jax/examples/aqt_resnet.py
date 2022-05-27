# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging

import jax
from jax import random
from jax import numpy as jnp

from absl import app
from absl import flags

from ml_collections import config_flags

from aqt.utils import hparams_utils as os_hparams_utils
from aqt.jax.imagenet import hparams_config
from aqt.jax.imagenet import input_pipeline
from aqt.jax.imagenet import models
from aqt.jax.imagenet import train_utils as imagenet_train_utils
from aqt.jax.imagenet.configs.paper import resnet50_w8_a8_auto

from iree.jax import Program, like, kernel

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file('hparams_config_dict',
                                resnet50_w8_a8_auto.__file__,
                                'Path to file defining a config dict.')


def main(argv):
  hparams = os_hparams_utils.load_hparams_from_config_dict(
      hparams_config.TrainingHParams, models.ResNet.HParams,
      FLAGS.hparams_config_dict)

  print("Instantiating model...")
  rng = random.PRNGKey(hparams.seed)
  batch_size = 1
  device_batch_size = 1
  image_size = 224
  model_dtype = jnp.float32
  model, variables = imagenet_train_utils.create_model(
      rng,
      device_batch_size,
      image_size,
      model_dtype,
      hparams=hparams.model_hparams,
      train=False,
      is_teacher=hparams.is_teacher)
  print("Model instantiated.")
  #print(f"  Model: {model}")
  #print(f"  Variables: {variables}")
  #print(f"  Quant context: {model.quant_context}")
  #model()

  example_input = jax.ShapedArray((1, 224, 224, 3), dtype=jnp.float32)

  class ResnetInferenceModel(Program):
    _variables = variables

    @kernel
    def _predict(v, image):
      # TODO: We should just be able to use mdl._variables here, but the
      # illusion is not good enough.
      logits = model.apply(v, image, mutable=False)
      return logits

    def predict(mdl, image=like(example_input)):
      return mdl._predict(mdl._variables, image)

  print(Program.get_mlir_module(ResnetInferenceModel))


if __name__ == "__main__":
  app.run(main)
