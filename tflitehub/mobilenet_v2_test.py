# RUN: %PYTHON %s

import absl.testing
import numpy
import test_util

model_path = "https://tfhub.dev/svampeatlas/lite-model/vision/classifier/fungi_mobile_V1/1/default/1?lite-format=tflite"

class MobilenetV2Test(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(MobilenetV2Test, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(MobilenetV2Test, self).compare_results(iree_results, tflite_results, details)
    self.assertTrue(numpy.isclose(iree_results[0], tflite_results[0], atol=1e-4).all())

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()

