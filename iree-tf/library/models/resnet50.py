import pathlib
import sys
import tensorflow as tf

# Add data python dir to the search path.
sys.path.insert(
    0,
    str(
        pathlib.Path(__file__).parent.parent.parent.parent / "data" /
        "python"))
from input_data import imagenet_test_data


# Input size 224x224x3 is used, as stated in the MLPerf Inference Rules: https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#41-benchmarks
class ResNet50(tf.Module):
    def __init__(self):
        super().__init__()
        self.model = tf.keras.applications.resnet50.ResNet50(
            weights="imagenet",
            include_top=True,
        )

    def generate_inputs(self, batch_size=1):
        image = imagenet_test_data.get_image_input()
        tensor = tf.convert_to_tensor(image)
        tensor = tf.image.convert_image_dtype(tensor, dtype=tf.float32)
        tensor = tf.keras.applications.resnet50.preprocess_input(tensor)
        tensor = tf.expand_dims(tensor, 0)
        tensor = tf.tile(tensor, [batch_size, 1, 1, 1])
        return (tensor,)

    @tf.function(jit_compile=True)
    def forward(self, inputs):
        return self.model(inputs, training=False)
