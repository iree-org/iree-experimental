# RUN: %PYTHON %s

import absl.testing
import imagenet_test_data
import numpy
import test_util

import os
import random
import re
from typing import Any, Callable, Mapping, Sequence, Set, Tuple, Union
import numpy as np

model_path = "https://storage.googleapis.com/tf_model_garden/vision/mobilenet/v2_1.0_int8/mobilenet_v2_1.00_224_int8.tflite"

def to_mlir_type(dtype: np.dtype) -> str:
  """Returns a string that denotes the type 'dtype' in MLIR style."""
  if not isinstance(dtype, np.dtype):
    # Handle np.int8 _not_ being a dtype.
    dtype = np.dtype(dtype)
  bits = dtype.itemsize * 8
  if np.issubdtype(dtype, np.integer):
    return f"i{bits}"
  elif np.issubdtype(dtype, np.floating):
    return f"f{bits}"
  else:
    raise TypeError(f"Expected integer or floating type, but got {dtype}")


def get_shape_and_dtype(array: np.ndarray,
                        allow_non_mlir_dtype: bool = False) -> str:
  shape_dtype = [str(dim) for dim in list(array.shape)]
  if np.issubdtype(array.dtype, np.number):
    shape_dtype.append(to_mlir_type(array.dtype))
  elif np.issubdtype(array.dtype, bool):
    shape_dtype.append("i8")
  elif allow_non_mlir_dtype:
    shape_dtype.append(f"<dtype '{array.dtype}'>")
  else:
    raise TypeError(f"Expected integer or floating type, but got {array.dtype}")
  return "x".join(shape_dtype)


def save_input_values(inputs: Sequence[np.ndarray],
                      file_path: str = None) -> str:
  result = []
  for array in inputs:
    shape_dtype = get_shape_and_dtype(array)
    if np.issubdtype(array.dtype, bool):
      values = 1 if array else 0
    else:
      values = " ".join([str(x) for x in array.flatten()])
    result.append(f"--function_input={shape_dtype}={values}")
  result = "\n".join(result)
  print("Saving IREE input values to: %s", file_path)
  with open(file_path, "w") as f:
    f.write(result)
    f.write("\n")
  return result

class MobilenetV2Int8Test(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(MobilenetV2Int8Test, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(MobilenetV2Int8Test, self).compare_results(iree_results, tflite_results, details)
    # Although this a quantized model, inputs and outputs are in float.
    # The difference here is quite high for a dequantized output.
    self.assertTrue(numpy.isclose(iree_results, tflite_results, atol=0.5).all())

    # Make sure the predicted class is the same.
    iree_predicted_class = numpy.argmax(iree_results[0][0])
    tflite_predicted_class = numpy.argmax(tflite_results[0][0])
    self.assertEqual(iree_predicted_class, tflite_predicted_class)

  def generate_inputs(self, input_details):
    inputs = imagenet_test_data.generate_input(self.workdir, input_details)
    # Normalize inputs to [-1, 1].
    inputs = (inputs.astype('float32') / 127.5) - 1
    save_input_values(inputs, '/tmp/iree-samples/inputs.txt')
    return [inputs]

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()

