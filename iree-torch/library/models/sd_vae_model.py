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

from diffusers import AutoencoderKL
from torchvision import transforms


# `VAE`: Variational auto-encoder.
# Compresses an input image into latent space using its encoder.
# Uncompresses latents into images using the decoder.
# Allows Stable Diffusion to perform diffusion in the latent space and convert
# to a higher resolution image using the `VAE` decoder.
class SDVaeModel(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.model = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae")

    def generate_inputs(self, batch_size=1):
        # We use VAE to encode an image and return this for input to the VAE decoder.
        image = imagenet_test_data.get_image_input().resize([512, 512])
        input = transforms.ToTensor()(image).unsqueeze(0)
        input = input.repeat(batch_size, 1, 1, 1)
        output = self.model.encode(input)
        sample = output.latent_dist.sample()
        return (sample, )

    def forward(self, input):
        return self.model.decode(input, return_dict=False)[0]
