# RUN: %PYTHON %s
# XFAIL: *

import absl.testing
import numpy
import test_util

model_path = "https://tfhub.dev/sayakpaul/lite-model/east-text-detector/dr/1?lite-format=tflite"

class EastTextDetectorTest(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(EastTextDetectorTest, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(EastTextDetectorTest, self).compare_results(iree_results, tflite_results, details)
    self.assertTrue(numpy.isclose(iree_results[0], tflite_results[0], atol=1e-3).all())
    self.assertTrue(numpy.isclose(iree_results[1], tflite_results[1], atol=1e-3).all())

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()




