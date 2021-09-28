
import absl.testing
import numpy
import test_util
import urllib.request

from PIL import Image

model_path = "https://tfhub.dev/agripredict/lite-model/disease-classification/1?lite-format=tflite"

class MobilenetQuantTest(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(MobilenetQuantTest, self).__init__(model_path, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(MobilenetQuantTest, self).compare_results(iree_results, tflite_results, details)
    self.assertTrue(numpy.isclose(iree_results[0], tflite_results[0], atol=1e-6).all())

  def generate_inputs(self, input_details):
    img_path = "https://storage.googleapis.com/tfhub-visualizers/agripredict/disease-classification/1/image_1_thumb.jpg"
    local_path = "/".join([self.workdir, "image.jpeg"])
    urllib.request.urlretrieve(img_path, local_path)

    shape = input_details[0]["shape"]
    im = numpy.array(Image.open(local_path).resize((shape[1], shape[2])))
    args = [im.reshape(shape)]
    return args

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()

