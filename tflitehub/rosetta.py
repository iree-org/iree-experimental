# RUN: %PYTHON %s
# XFAIL: *

import absl.testing
import numpy
import test_util
import unittest

model_path = "https://tfhub.dev/tulasiram58827/lite-model/rosetta/dr/1?lite-format=tflite"

# Failing due to tfl.split failure
class RosettaTest(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(RosettaTest, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(RosettaTest, self).compare_results(iree_results, tflite_results, details)
    self.assertTrue(numpy.isclose(iree_results[0], tflite_results[0], atol=5e-3).all())

  @unittest.expectedFailure
  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()

