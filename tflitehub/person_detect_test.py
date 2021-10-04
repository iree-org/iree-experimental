# RUN: %PYTHON %s
# XFAIL: *

import absl
import absl.testing
import iree.compiler.tflite as iree_tflite_compile
import iree.runtime as iree_rt
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

  def compile_and_execute(self):
    '''temporarily bypass tflite execute'''
    absl.logging.info("Setting up for IREE")
    iree_tflite_compile.compile_file(
      self.tflite_file, input_type="tosa",
      output_file=self.binary,
      save_temp_tfl_input=self.tflite_ir,
      save_temp_iree_input=self.iree_ir,
      target_backends=iree_tflite_compile.DEFAULT_TESTING_BACKENDS,
      import_only=False)

    input_details = [{"shape": [1, 96, 96, 1]}]
    absl.logging.info("Setting up test inputs")
    args = self.generate_inputs(input_details)

    absl.logging.info("Invoke IREE")
    iree_results = None
    with open(self.binary, 'rb') as f:
      config = iree_rt.Config("dylib")
      ctx = iree_rt.SystemContext(config=config)
      vm_module = iree_rt.VmModule.from_flatbuffer(f.read())
      ctx.add_vm_module(vm_module)
      invoke = ctx.modules.module["main"]
      iree_results = invoke(*args)
      if not isinstance(iree_results, tuple):
        iree_results = (iree_results,)
    output_details = [{"dtype": "int8"}]
    # Get the tflite result from
    # https://github.com/tensorflow/tflite-micro/blob/main/tensorflow/lite/micro/examples/person_detection/person_detection_test.cc
    tflite_results = numpy.array([-113, 113], dtype=numpy.int8)
    tflite_results = [tflite_results.reshape([1, 2])]
    self.compare_results(iree_results, tflite_results, output_details)

  def test_compile_tflite(self):
    # tflite.interpreter python API has problem rendering this file. Issue filed.
    # The example would fail after the iree_tflite_compile.compile_file.
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()
