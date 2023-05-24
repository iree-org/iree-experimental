import pathlib
import sys
import torch

# Add data python dir to the search path.
sys.path.insert(
    0,
    str(
        pathlib.Path(__file__).parent.parent.parent.parent / "data" /
        "python"))
from input_data import imagenet_test_data
from torchvision.models import resnet50, ResNet50_Weights


# We use the ResNet50 variant listed in MLPerf here: https://github.com/mlcommons/inference/tree/master/vision/classification_and_detection
# Input size 224x224x3 is used, as stated in the MLPerf Inference Rules: https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#41-benchmarks
class Resnet50(torch.nn.Module):

    def __init__(self):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT
        self.model = resnet50(weights=weights)
        self.preprocess = weights.transforms()
        self.train(False)

    def generate_inputs(self, batch_size=1, dtype=torch.float32):
        image = imagenet_test_data.get_image_input()
        tensor = self.preprocess(image).to(dtype=dtype).unsqueeze(0)
        tensor = tensor.repeat(batch_size, 1, 1, 1)
        return (tensor, )

    def forward(self, input):
        return self.model(input)
