# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Usage: python run_trainer.py /path/to/mnist_train.vmfb [/path/to/checkpoint.npz]

import os
import shutil
import sys
from iree import runtime as iree_rt

import numpy.random as npr
import numpy as np
from examples import datasets


def main(args):
  vmfb_file = args[0]
  checkpoint_file = None
  if len(args) > 1:
    checkpoint_file = args[1]
  config = iree_rt.system_api.Config("local-task")
  trainer_module = iree_rt.system_api.load_vm_flatbuffer_file(vmfb_file,
                                                              driver="local-task")
  print(trainer_module)

  if checkpoint_file and os.path.exists(checkpoint_file):
    print(f"Loading checkpoint {checkpoint_file}")
    loaded_arrays = []
    with np.load(checkpoint_file) as checkpoint:
      for _, value in checkpoint.items():
        loaded_arrays.append(value)
    trainer_module.set_opt_state(*loaded_arrays)
  else:
    print("Random initializing...")
    trainer_module.initialize(np.asarray([232, 843], dtype=np.int32))

  print("Stepping...")
  train_batch = get_examples()
  print(trainer_module.update)
  i = 0
  while True:
    i += 1
    batch = next(train_batch)
    trainer_module.update(*batch)
    accuracy = compute_accuracy(batch, trainer_module)
    if (i % 100) == 0:
      print(f"Step {i} accuracy = {accuracy}")
    if checkpoint_file:
      save_checkpoint(checkpoint_file, *trainer_module.get_opt_state())


def compute_accuracy(batch, trainer_module):
  inputs, targets = batch
  prediction_posterior = trainer_module.predict(inputs)
  target_class = np.argmax(targets, axis=1)
  predicted_class = np.argmax(prediction_posterior, axis=1)
  return np.mean(predicted_class == target_class)


def get_examples():
  batch_size = 128
  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, batch_size)
  print(f"Number of batches in dataset: {num_complete_batches}")
  def data_stream():
    rng = npr.RandomState(0)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_complete_batches):
        batch_idx = perm[i * batch_size:(i + 1) * batch_size]
        batch = train_images[batch_idx], train_labels[batch_idx]
        assert batch[0].shape[0] == batch_size
        yield batch

  batches = data_stream()
  return batches


def save_checkpoint(checkpoint_file, *arrays):
  temp_file = os.path.join(os.path.dirname(checkpoint_file),
                           "_tmp_" + os.path.basename(checkpoint_file))
  np.savez(temp_file, *arrays)
  shutil.move(temp_file, checkpoint_file)


if __name__ == "__main__":
  main(sys.argv[1:])
