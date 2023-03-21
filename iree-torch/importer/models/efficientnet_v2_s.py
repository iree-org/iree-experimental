import torch

from models.input_data import imagenet_test_data
from torchvision.models import efficientnet_v2_s, EfficientNet_V2_S_Weights


# EfficientNet-V2-Small has input image size of 384x384.
class EfficientNetV2S(torch.nn.Module):

    def __init__(self):
        super().__init__()
        weights = EfficientNet_V2_S_Weights.DEFAULT
        self.model = efficientnet_v2_s(weights=weights)
        self.preprocess = weights.transforms()
        self.train(False)

    def generate_inputs(self, batch_size=1):
        image = imagenet_test_data.get_image_input()
        tensor = self.preprocess(image).unsqueeze(0)
        tensor = tensor.repeat(batch_size, 1, 1, 1)
        return (tensor, )

    def forward(self, input):
        return self.model(input)
