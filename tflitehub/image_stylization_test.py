# RUN: %PYTHON %s
# XFAIL: *

import absl.testing
import test_util

model_path = "https://tfhub.dev/sayakpaul/lite-model/arbitrary-image-stylization-inceptionv3-dynamic-shapes/int8/predict/1?lite-format=tflite"

# Failure is due to avg_pool2d.
class ImageStylizationTest(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(ImageStylizationTest, self).__init__(model_path, *args, **kwargs)

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()
