import absl.testing
import numpy
import test_util
import urllib.request

from PIL import Image

model_path = "https://github.com/tensorflow/tflite-micro/raw/main/tensorflow/lite/micro/models/person_detect.tflite"

class PersonDetectTest(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(PersonDetectTest, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(PersonDetectTest, self).compare_results(iree_results, tflite_results, details)
    self.assertTrue(numpy.isclose(iree_results[0], tflite_results[0], atol=2).all())

  def generate_inputs(self, input_details):
    img_path = "https://github.com/tensorflow/tflite-micro/raw/main/tensorflow/lite/micro/examples/person_detection/testdata/person.bmp"
    local_path = "/".join([self.workdir, "person.bmp"])
    urllib.request.urlretrieve(img_path, local_path)

    shape = input_details[0]["shape"]
    im = numpy.array(Image.open(local_path).resize((shape[1], shape[2])))
    args = [im.reshape(shape)]
    return args

  def test_compile_tflite(self):
    # tflite.interpreter python API has problem rendering this file. Issue filed.
    # The example would fail after the iree_tflite_compile.compile_file.
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()
