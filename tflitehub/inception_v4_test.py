# RUN: %PYTHON %s

import absl.testing
import test_util

model_path = "https://tfhub.dev/tensorflow/lite-model/inception_v4/1/metadata/1?lite-format=tflite"

class InceptionV4Test(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(InceptionV4Test, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(InceptionV4Test, self).compare_results(iree_results, tflite_results, details)

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()




