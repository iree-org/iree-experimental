
import absl.testing
import test_util

model_path = "https://tfhub.dev/tensorflow/lite-model/mobilenet_v1_1.0_160/1/default/1?lite-format=tflite"

class PoseTest(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(PoseTest, self).__init__(model_path, *args, **kwargs)

  def test_compile_tflite(self):
    self.compile_and_execute() 

if __name__ == '__main__':
  absl.testing.absltest.main()

