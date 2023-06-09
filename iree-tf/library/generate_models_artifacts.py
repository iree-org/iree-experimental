import argparse
import numpy as np
import os
import tensorflow as tf

from models import resnet50, bert_large, t5_large
from multiprocessing import Process

_MODEL_NAME_TO_MODEL_CONFIG = {
    # Batch sizes taken from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/resnet50
    "RESNET50": (resnet50.ResNet50, [1, 8, 64, 128, 256, 2048]),
    # Batch sizes based on MLPerf config: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/bert
    "BERT_LARGE": (bert_large.BertLarge, [1, 16, 24, 32, 48, 64, 512, 1024, 1280]),
    # Uses the same batch sizes as Bert-Large.
    "T5_LARGE": (t5_large.T5Large, [1, 16, 24, 32, 48, 64, 512]),
}

def generate_artifacts(model_name: str, model_class: type[tf.Module], batch_size: int, save_dir: str):
    try:
        # Configure to dump hlo.
        hlo_dir = os.path.join(save_dir, "hlo")
        os.makedirs(save_dir, exist_ok=True)
        # Only dump hlo for forward function.
        os.environ["XLA_FLAGS"] = f"--xla_dump_to={hlo_dir} --xla_dump_hlo_module_re=.*inference_forward.*"

        model = model_class()
        inputs = model.generate_inputs(batch_size)

        # Save inputs.
        for idx, input in enumerate(inputs):
            input_path = os.path.join(save_dir, f"input_{idx}.npy")
            print(f"Saving input {input.get_shape()} to {input_path}")
            np.save(input_path, input)

        # Save output.
        outputs = model.forward(*inputs)
        output_path = os.path.join(save_dir, f"output_0.npy")
        print(f"Saving output {outputs.get_shape()} to {output_path}")
        np.save(output_path, outputs)

        # Save saved model.
        tensor_specs = []
        for input in inputs:
            tensor_specs.append(tf.TensorSpec.from_tensor(input))
        call_signature = model.forward.get_concrete_function(*tensor_specs)

        saved_model_path = os.path.join(save_dir, "saved_model")
        os.makedirs(saved_model_path, exist_ok=True)
        print(f"Saving {saved_model_path} with call signature: {call_signature}")
        tf.saved_model.save(model, saved_model_path,
                            signatures={"serving_default": call_signature})
    except Exception as e:
        print(f"Failed to import model {model_name}. Exception: {e}")


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-o",
                           "--output_dir",
                           default="/tmp",
                           help="Path to save mlir files")
    args = argParser.parse_args()

    for model_name, model_config in _MODEL_NAME_TO_MODEL_CONFIG.items():
        print(f"\n\n--- {model_name} -------------------------------------")
        model_class, batch_sizes = model_config

        model_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        for batch_size in batch_sizes:
            save_dir = os.path.join(model_dir, f"batch_{batch_size}")
            os.makedirs(save_dir, exist_ok=True)

            p = Process(target=generate_artifacts,
                        args=(model_name, model_class, batch_size, save_dir))
            p.start()
            p.join()
