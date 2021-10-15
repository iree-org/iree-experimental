# RUN: %PYTHON %s

import absl.testing
import numpy
import test_util
import urllib

from PIL import Image

model_path = "https://tfhub.dev/tensorflow/lite-model/efficientnet/lite3/int8/2?lite-format=tflite"

class EfficientNetTest(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(EfficientNetTest, self).__init__(model_path, *args, **kwargs)

  def generate_inputs(self, input_details):
    img_path = "https://github.com/google-coral/test_data/raw/master/cat.bmp"
    local_path = "/".join([self.workdir, "cat.bmp"])
    urllib.request.urlretrieve(img_path, local_path)

    shape = input_details[0]["shape"]
    im = numpy.array(Image.open(local_path).resize((shape[1], shape[2])))
    args = [im.reshape(shape)]
    return args

  def compare_results(self, iree_results, tflite_results, details):
    super(EfficientNetTest, self).compare_results(iree_results, tflite_results, details)
    iree = iree_results[0].flatten().astype(numpy.single) 
    tflite = tflite_results[0].flatten().astype(numpy.single)
    self.assertTrue(numpy.isclose(iree, tflite, atol=8).all())

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()

