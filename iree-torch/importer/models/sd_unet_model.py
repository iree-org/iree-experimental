import torch

from diffusers import UNet2DConditionModel
from models import sd_clip_text_model


# Consists of `ResNet` encoder and decoder blocks with cross-attention layers.
# Used in Stable Diffusion to gradually subtract noise in the latent space.
# Usually run over multiple steps until the image is sufficiently de-noised.
class SDUnetModel(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.model = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet",
    )
    self.train(False)

  def generate_inputs(self):
    # Use `SDClipTextModel` to generate text embeddings.
    clip = sd_clip_text_model.SDClipTextModel()
    text_embedding = clip.forward(*clip.generate_inputs())

    # Use a random noise latent input.
    latents = torch.rand([
        1, self.model.in_channels, self.model.config.sample_size,
        self.model.config.sample_size
    ])
    return (latents, text_embedding)

  def forward(self, latents, text_embedding):
    timestep = 100
    return self.model(latents, timestep,
                      encoder_hidden_states=text_embedding)[0]
