# RUN: %PYTHON %s

import absl.testing
import numpy
import test_util
import unittest

model_path = "https://storage.googleapis.com/iree-model-artifacts/mobilenet_v3-large_224_1.0_float.tflite"

class MobilenetV3LargeTest(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(MobilenetV3LargeTest, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(MobilenetV3LargeTest, self).compare_results(iree_results, tflite_results, details)
    self.assertTrue(numpy.isclose(iree_results[0], tflite_results[0], atol=1e-4).all())

  @unittest.expectedFailure
  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()

