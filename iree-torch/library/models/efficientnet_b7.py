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
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights


# EfficientNetB7 has input image size of 600x600.
class EfficientNetB7(torch.nn.Module):

    def __init__(self):
        super().__init__()
        weights = EfficientNet_B7_Weights.DEFAULT
        self.model = efficientnet_b7(weights=weights)
        self.preprocess = weights.transforms()
        self.train(False)

    def generate_inputs(self, batch_size=1, dtype=torch.float32):
        assert dtype == torch.float32, "Input generation only implemented for float32"

        image = imagenet_test_data.get_image_input()
        tensor = self.preprocess(image).unsqueeze(0)
        tensor = tensor.repeat(batch_size, 1, 1, 1)
        return (tensor, )

    def forward(self, input):
        return self.model(input)
