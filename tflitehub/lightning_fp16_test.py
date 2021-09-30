# RUN: %PYTHON %s
# XFAIL: *

import absl.testing
import test_util
import unittest

model_path = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite"

# Currently failing further in the linalg stack:
#   Tosa does not support fp16, we need to find a workaround.
class LightningFp16Test(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(LightningFp16Test, self).__init__(model_path, *args, **kwargs)

  @unittest.expectedFailure
  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()
