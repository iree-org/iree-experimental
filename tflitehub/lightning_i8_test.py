import absl.testing
import numpy
import test_util

model_path = "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite"

# Currently failing further in the linalg stack:
#   Invalid cast from ui8 to f32 TODO: make tfl.cast insert a rescale for ui8
class LightningI8Test(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(LightningI8Test, self).__init__(model_path, *args, **kwargs)

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()
