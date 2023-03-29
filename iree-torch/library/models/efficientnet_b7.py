import torch

from models.input_data import imagenet_test_data
from torchvision.models import efficientnet_b7, EfficientNet_B7_Weights


# EfficientNetB7 has input image size of 600x600.
class EfficientNetB7(torch.nn.Module):

    def __init__(self):
        super().__init__()
        weights = EfficientNet_B7_Weights.DEFAULT
        self.model = efficientnet_b7(weights=weights)
        self.preprocess = weights.transforms()
        self.train(False)

    def generate_inputs(self, batch_size=1):
        image = imagenet_test_data.get_image_input()
        tensor = self.preprocess(image).unsqueeze(0)
        tensor = tensor.repeat(batch_size, 1, 1, 1)
        return (tensor, )

    def forward(self, input):
        return self.model(input)
