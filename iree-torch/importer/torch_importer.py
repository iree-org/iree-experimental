import argparse
import numpy as np
import os
import torch
import torch_mlir

from import_utils import import_torch_module_with_fx
from models import sd_clip_text_model, sd_vae_model, sd_unet_model

_MODEL_NAME_TO_MODEL_CLASS = {
    "SD_CLIP_TEXT_MODEL_SEQLEN64": sd_clip_text_model.SDClipTextModel,
    "SD_VAE_MODEL": sd_vae_model.SDVaeModel,
    "SD_UNET_MODEL": sd_unet_model.SDUnetModel
}


def import_to_mlir(model):
  inputs = model.generate_inputs()
  mlir_data = import_torch_module_with_fx(
      model, inputs, torch_mlir.OutputType.LINALG_ON_TENSORS)
  return (inputs, mlir_data)


if __name__ == "__main__":
  argParser = argparse.ArgumentParser()
  argParser.add_argument("-o",
                         "--output_dir",
                         default="/tmp",
                         help="Path to save mlir files")
  args = argParser.parse_args()

  for model_name, model_class in _MODEL_NAME_TO_MODEL_CLASS.items():
    print(f"\n\n--- {model_name} -------------------------------------")
    model_dir = os.path.join(args.output_dir, model_name)
    os.mkdir(model_dir)

    try:
      # Remove all gradient info from models and tensors since these models are inference only.
      with torch.no_grad():
        model = model_class()
        inputs, mlir_data = import_to_mlir(model)

        # Save inputs.
        for idx, input in enumerate(inputs):
          input_path = os.path.join(model_dir, f"input_{idx}.npy")
          print(f"Saving input {idx} at {input_path}")
          np.save(input_path, input)

        # Save mlir.
        save_path = os.path.join(model_dir, "linalg.mlir")
        with open(save_path, "wb") as mlir_file:
          mlir_file.write(mlir_data)
          print(f"Saved {save_path}")
    except Exception as e:
      print(f"Failed to import model {model_name}. Exception: {e}")
