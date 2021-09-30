import absl.flags
import absl.testing
import numpy
import test_util

absl.flags.DEFINE_string('model', None, 'model path to execute')

# Flag based invocation, useful for trying new models.
class ManualTest(test_util.TFLiteModelTest):
  def __init__(self, *args, **kwargs):
    super(ManualTest, self).__init__(absl.flags.FLAGS.model, *args, **kwargs)

  def compare_results(self, iree_results, tflite_results, details):
    super(ManualTest, self).compare_results(iree_results, tflite_results, details)

  def test_compile_tflite(self):
    self.compile_and_execute()

if __name__ == '__main__':
  absl.testing.absltest.main()

