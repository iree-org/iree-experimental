import torch

from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms
import urllib.request


# `VAE`: Variational auto-encoder.
# Compresses an input image into latent space using its encoder.
# Uncompresses latents into images using the decoder.
# Allows Stable Diffusion to perform diffusion in the latent space and convert
# to a higher resolution image using the `VAE` decoder.
class SDVaeModel(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.model = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4",
                                               subfolder="vae")

  def generate_inputs(self):
    # We use VAE to encode an image and return this for input to the VAE decoder.
    img_path = "https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG"
    local_path = "/tmp/ILSVRC2012_val_00000023.JPEG"
    urllib.request.urlretrieve(img_path, local_path)
    image = Image.open(local_path).convert("RGB").resize([512, 512])
    input = transforms.ToTensor()(image).unsqueeze(0)
    output = self.model.encode(input)
    sample = output.latent_dist.sample()
    return (sample.unsqueeze(0))

  def forward(self, input):
    return self.model.decode(input, return_dict=False)[0]
