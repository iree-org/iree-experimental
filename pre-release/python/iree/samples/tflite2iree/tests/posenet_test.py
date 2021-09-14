
import absl.testing
import test_util

model_path = "https://storage.googleapis.com/download.tensorflow.org/models/tflite/gpu/multi_person_mobilenet_v1_075_float.tflite"

class PoseTest(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(PoseTest, self).__init__(model_path, *args, **kwargs)

  def test_compile_tflite(self):
    self.compile_and_execute() 

if __name__ == '__main__':
  absl.testing.absltest.main()

