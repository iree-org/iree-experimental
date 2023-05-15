import argparse
import numpy as np
import os
import torch
import torch_mlir

from import_utils import import_torch_module_with_fx, import_torch_module
from models import bert_large, sd_clip_text_model, sd_unet_model, sd_vae_model, resnet50, t5_large, efficientnet_b7, efficientnet_v2_s

from dataclasses import dataclass
from typing import Iterable, Dict

@dataclass
class ModelConfig:
    model_class: torch.nn.Module
    batch_sizes: Iterable[int]
    dtype: torch.dtype = torch.float32
    force_gpu: bool = False
    import_with_fx: bool = True         # Determines use of import_torch_module_with_fx vs import_torch_module


_MODEL_NAME_TO_MODEL_CONFIG: Dict[str, ModelConfig] = {
    # Batch sizes based on MLPerf config: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/bert
    "BERT_LARGE":
    ModelConfig(bert_large.BertLarge, [1, 16, 24, 32, 48, 64, 512, 1024, 1280]),
    "EFFICIENTNET_B7": ModelConfig(efficientnet_b7.EfficientNetB7, [
        1,
    ]),
    "EFFICIENTNET_V2_S": ModelConfig(efficientnet_v2_s.EfficientNetV2S, [
        1,
    ]),
    # Batch sizes based on MLPerf config: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/resnet50
    "RESNET50": ModelConfig(resnet50.Resnet50, [1, 8, 64, 128, 256, 2048]),
    "SD_CLIP_TEXT_MODEL_SEQLEN64": ModelConfig(sd_clip_text_model.SDClipTextModel, [
        1,
    ]),
    "SD_VAE_MODEL": ModelConfig(sd_vae_model.SDVaeModel, [
        1,
    ]),
    "SD_UNET_MODEL": ModelConfig(sd_unet_model.SDUnetModel, [
        1,
    ]),
    "T5_LARGE": ModelConfig(t5_large.T5Large, [1,]),

    # FP16 models below
    "BERT_LARGE_FP16": ModelConfig(bert_large.BertLarge, 
                                   [1, 16, 24, 32, 48, 64, 512, 1024, 1280], 
                                   torch.float16, 
                                   force_gpu=True),
    "EFFICIENTNET_V2_S_FP16": ModelConfig(efficientnet_v2_s.EfficientNetV2S, [
        1,
    ], torch.float16, force_gpu=True, import_with_fx=False),
    "RESNET50_FP16": ModelConfig(resnet50.Resnet50, 
                                 [1, 8, 64, 128, 256, 2048], 
                                 torch.float16, 
                                 force_gpu=True, 
                                 import_with_fx=False),
}


def import_to_mlir(model_config: ModelConfig, batch_size: int):
    model = model_config.model_class().to(dtype=model_config.dtype)
    orig_inputs = model.generate_inputs(batch_size=batch_size, dtype=model_config.dtype)
    converted_inputs = orig_inputs

    if model_config.force_gpu:
        model.cuda()
        converted_inputs = [input.cuda() for input in orig_inputs]
    
    if model_config.import_with_fx:
        mlir_data = import_torch_module_with_fx(
            model, converted_inputs, torch_mlir.OutputType.LINALG_ON_TENSORS)
    else:
        graph = torch.jit.trace(model, converted_inputs)
        mlir_data = import_torch_module(graph, converted_inputs, torch_mlir.OutputType.LINALG_ON_TENSORS)
        
        
    outputs = model.forward(*converted_inputs)
    outputs = outputs.cpu()
    return (orig_inputs, outputs,  mlir_data)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-o",
                           "--output_dir",
                           default="/tmp",
                           help="Path to save mlir files")
    args = argParser.parse_args()

    for model_name, model_config in _MODEL_NAME_TO_MODEL_CONFIG.items():
        print(f"\n\n--- {model_name} -------------------------------------")
        if model_config.force_gpu and not torch.cuda.is_available():
            print("SKIPPED due to missing CUDA installation")
            continue

        model_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        for batch_size in model_config.batch_sizes:
            save_dir = os.path.join(model_dir, f"batch_{batch_size}")
            os.makedirs(save_dir, exist_ok=True)

            try:
                # Remove all gradient info from models and tensors since these models are inference only.
                with torch.no_grad():
                    inputs, outputs, mlir_data = import_to_mlir(model_config, batch_size)

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
                    output_path = os.path.join(save_dir, f"output_0.npy")
                    print(f"Saving output 0 to {output_path}")
                    np.save(output_path, outputs.cpu())

            except Exception as e:
                print(f"Failed to import model {model_name}. Exception: {e}")
