# RUN: %PYTHON %s %config_flag
# TODO(iree/#13488): Remove the XFAIL
# XFAIL: vmvx

import absl.testing
import numpy as np
import squad_test_data
import test_util

model_path = "https://storage.googleapis.com/iree-model-artifacts/mobilebert-baseline-tf2-quant.tflite"

class MobileBertTest(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(MobileBertTest, self).__init__(model_path, *args, **kwargs)

  # Inputs modified to be useful mobilebert inputs.
  def generate_inputs(self, input_details):
    for input in input_details:
      absl.logging.info("\t%s, %s", str(input["shape"]), input["dtype"].__name__)

    input_0 = np.asarray(squad_test_data._INPUT_WORD_ID, dtype=input_details[0]["dtype"])
    input_1 = np.asarray(squad_test_data._INPUT_TYPE_ID, dtype=input_details[1]["dtype"])
    input_2 = np.asarray(squad_test_data._INPUT_MASK, dtype=input_details[2]["dtype"])
    return [
        input_0.reshape(input_details[0]["shape"]),
        input_1.reshape(input_details[1]["shape"]),
        input_2.reshape(input_details[2]["shape"])
    ]

  def compare_results(self, iree_results, tflite_results, details):
    super(MobileBertTest, self).compare_results(iree_results, tflite_results, details)
    # We have confirmed in large scale accuracy tests that differences this large is acceptable.
    self.assertTrue(np.isclose(iree_results[0], tflite_results[0], atol=7.0).all())
    self.assertTrue(np.isclose(iree_results[1], tflite_results[1], atol=7.0).all())

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()
