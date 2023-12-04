import jax.numpy as jnp
import pathlib
import sys

from transformers import AutoImageProcessor, FlaxResNetModel

# Add data python dir to the search path.
sys.path.insert(
    0,
    str(
        pathlib.Path(__file__).parent.parent.parent.parent / "data" /
        "python"))
from input_data import imagenet_test_data


# Input size 224x224x3 is used, as stated in the MLPerf Inference Rules: https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#41-benchmarks
class ResNet50():
    def __init__(self, dtype=jnp.float32):
        super().__init__()
        self.model = FlaxResNetModel.from_pretrained("microsoft/resnet-50", dtype=dtype)
        if dtype == jnp.float16:
            self.model.params = self.model.to_fp16(self.model.params)
        elif dtype == jnp.bfloat16:
            self.model.params = self.model.to_bf16(self.model.params)

    def generate_inputs(self, batch_size=1):
        image = imagenet_test_data.get_image_input()
        image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
        inputs = image_processor(images=image, return_tensors="jax")
        tensor = inputs["pixel_values"]
        tensor = jnp.asarray(jnp.tile(tensor, [batch_size, 1, 1, 1]), dtype=self.model.dtype)
        return (tensor,)

    def forward(self, inputs):
        return self.model(inputs)[0]
