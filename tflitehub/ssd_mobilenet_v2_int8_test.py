# RUN: %PYTHON %s
# XFAIL: *

import absl.testing
import numpy
import test_util

model_path = "https://storage.googleapis.com/iree-model-artifacts/ssd_mobilenet_v2_dynamic_1.0_int8.tflite"

class SsdMobilenetV2Test(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(SsdMobilenetV2Test, self).__init__(model_path, *args, **kwargs)

  def generate_inputs(self, input_details):
    img_path = "https://github.com/tensorflow/examples/raw/master/lite/examples/pose_estimation/raspberry_pi/test_data/image3.jpeg"
    local_path = "/".join([self.workdir, "person.jpg"])
    urllib.request.urlretrieve(img_path, local_path)

    shape = input_details[0]["shape"]
    im = numpy.array(Image.open(local_path).resize((shape[1], shape[2])))
    args = [im.reshape(shape)]
    return args

  def compare_results(self, iree_results, tflite_results, details):
    super(SsdMobilenetV1Test, self).compare_results(iree_results, tflite_results, details)
    for i in range(len(iree_results)):
      self.assertTrue(numpy.isclose(iree_results[i], tflite_results[i], atol=1e-4).all())

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()

