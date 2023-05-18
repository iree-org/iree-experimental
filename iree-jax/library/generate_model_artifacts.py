import argparse
import jax
import jax.numpy as jnp
import numpy as np
import os

from models import bert_large
from multiprocessing import Process
from typing import Any

_MODEL_NAME_TO_MODEL_CONFIG = {
    # Batch sizes based on MLPerf config: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/bert
    "BERT_LARGE":
        (bert_large.BertLarge, [1, 16, 24, 32, 48, 64, 512, 1024, 1280]),
}


def generate_artifacts(model_name: str, model_class: Any, batch_size: int,
                       save_dir: str):
  try:
    # Configure to dump hlo.
    hlo_dir = os.path.join(save_dir, "hlo")
    os.makedirs(save_dir, exist_ok=True)
    # Only dump hlo for the inference function `jit_model_jitted`.
    os.environ[
        "XLA_FLAGS"] = f"--xla_dump_to={hlo_dir} --xla_dump_hlo_module_re=.*jit_forward.*"

    model = model_class()
    inputs = model.generate_inputs(batch_size)

    jit_inputs = jax.device_put(inputs)
    jit_function = jax.jit(model.forward)

    # Save inputs.
    for idx, input in enumerate(inputs):
      input_path = os.path.join(save_dir, f"input_{idx}.npy")
      print(f"Saving input {jnp.shape(input)} to {input_path}")
      np.save(input_path, input)

    # Save output.
    outputs = jit_function(*jit_inputs)
    output_path = os.path.join(save_dir, f"output_0.npy")
    print(f"Saving output {jnp.shape(outputs)} to {output_path}")
    np.save(output_path, outputs)

    # Export.
    mlir = jit_function.lower(*jit_inputs).compiler_ir(dialect="stablehlo")
    with open(f"{save_dir}/stablehlo.mlir", "w") as f:
      f.write(str(mlir))

  except Exception as e:
    print(f"Failed to import model {model_name}. Exception: {e}")
    raise


if __name__ == "__main__":
  argParser = argparse.ArgumentParser()
  argParser.add_argument("-o",
                         "--output_dir",
                         default="/tmp",
                         help="Path to save model artifacts")
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
