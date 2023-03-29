import argparse
import numpy as np
import os
import torch
import torch_mlir

from import_utils import import_torch_module_with_fx
from models import bert_large, sd_clip_text_model, sd_unet_model, sd_vae_model, resnet50, t5_large, efficientnet_b7, efficientnet_v2_s

_MODEL_NAME_TO_MODEL_CONFIG = {
    "BERT_LARGE":
    (bert_large.BertLarge, [1, 8, 16, 32, 64, 128, 256, 512, 1024]),
    "EFFICIENTNET_B7": (efficientnet_b7.EfficientNetB7, [
        1,
    ]),
    "EFFICIENTNET_V2_S": (efficientnet_v2_s.EfficientNetV2S, [
        1,
    ]),
    "RESNET50": (resnet50.Resnet50, [1, 8, 16, 32, 64, 128, 256, 512, 1024]),
    "SD_CLIP_TEXT_MODEL_SEQLEN64": (sd_clip_text_model.SDClipTextModel, [
        1,
    ]),
    "SD_VAE_MODEL": (sd_vae_model.SDVaeModel, [
        1,
    ]),
    "SD_UNET_MODEL": (sd_unet_model.SDUnetModel, [
        1,
    ]),

    #"T5_LARGE": (t5_large.T5Large, [1, 8, 16, 32, 64, 128, 256, 512, 1024]),
}


def import_to_mlir(model, batch_size):
    inputs = model.generate_inputs(batch_size=batch_size)
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

    for model_name, model_config in _MODEL_NAME_TO_MODEL_CONFIG.items():
        print(f"\n\n--- {model_name} -------------------------------------")
        model_class = model_config[0]
        batch_sizes = model_config[1]

        model_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        for batch_size in batch_sizes:
            save_dir = os.path.join(model_dir, f"batch_{batch_size}")
            os.makedirs(save_dir, exist_ok=True)

            try:
                # Remove all gradient info from models and tensors since these models are inference only.
                with torch.no_grad():
                    model = model_class()
                    inputs, mlir_data = import_to_mlir(model, batch_size)

                    # Save inputs.
                    for idx, input in enumerate(inputs):
                        input_path = os.path.join(save_dir, f"input_{idx}.npy")
                        print(f"Saving input {idx} to {input_path}")
                        np.save(input_path, input)

                    # Save mlir.
                    save_path = os.path.join(save_dir, "linalg.mlir")
                    with open(save_path, "wb") as mlir_file:
                        mlir_file.write(mlir_data)
                        print(f"Saved {save_path}")

                    # Save output.
                    outputs = model.forward(*inputs)
                    output_path = os.path.join(save_dir, f"output_0.npy")
                    print(f"Saving output 0 to {output_path}")
                    np.save(output_path, outputs)

            except Exception as e:
                print(f"Failed to import model {model_name}. Exception: {e}")
