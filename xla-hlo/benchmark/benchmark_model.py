import argparse
import json
import pathlib
import re
import requests
import statistics
import subprocess
import sys

from typing import Optional

# Add benchmark definitions to the search path.
sys.path.insert(
    0,
    str(
        pathlib.Path(__file__).parent.parent.parent / "oobi" /
        "benchmark-definitions" / "python"))
import data_types, jax_model_definitions, model_dictionary, tf_model_definitions, unique_ids
from utils import execution_environment


def benchmark_lookup(unique_id: str):
  if unique_id not in model_dictionary.MODEL_DICT:
    id_list = '\n  '.join(model_dictionary.MODEL_DICT.keys())
    raise ValueError(f"Id {unique_id} does not exist in model suite. Expected "
                     f"one of:\n  {id_list}")

  return model_dictionary.MODEL_DICT[unique_id]


def dump_result(file_path: str, result: dict) -> None:
  with open(file_path, "r") as f:
    dictObj = json.load(f)

  dictObj["benchmarks"].append(result)

  with open(file_path, "w") as f:
    json.dump(dictObj, f)


def bytes_to_mb(bytes: Optional[int]) -> Optional[float]:
  return None if bytes is None else bytes / 1e6


def run_compiler_benchmark(hlo_benchmark_tool_path: str, hlo_input_path: str,
                           benchmark_iterations: int, device: str) -> dict:
  cmd = [
      hlo_benchmark_tool_path,
      "--input_format=hlo",
      f"--platform={device}",
      "--reference_platform=",
      "--logtostderr",
      f"--input_module={hlo_input_path}",
      f"--iterations={benchmark_iterations}",
  ]
  result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  result_text = result.stdout.decode("utf-8")

  regex = re.compile(r"... compiled and ran in (.*)s.")
  matches = re.findall(regex, result_text)
  # Take the first iteration compile-time latency. Profiles show that this is
  # where tuning and other initialization occurs. Subsequent calls to compile
  # in the same process will reuse these results.
  compile_time_latency = float(matches[0]) if matches else None

  regex = re.compile(r"execution time for runner [A-Za-z]*: (.*)s.")
  matches = re.findall(regex, result_text)
  assert len(matches) == benchmark_iterations, (
      f"Expected to find {benchmark_iterations} latencies but found "
      f"{len(matches)} instead:\n{result_text}")
  latencies = [float(match) * 1000 for match in matches]

  results_dict = {
      "compile_time_s": compile_time_latency,
      "min_latency_ms": min(latencies, default=None),
      "max_latency_ms": max(latencies, default=None),
      "mean_latency_ms": statistics.mean(latencies) if latencies else None,
      "median_latency_ms": statistics.median(latencies) if latencies else None,
      "stddev_latency_ms": statistics.stdev(latencies) if latencies else None,
      "benchmark_iterations": benchmark_iterations,
  }
  return results_dict


if __name__ == "__main__":
  argParser = argparse.ArgumentParser()
  argParser.add_argument(
      "-o",
      "--output_path",
      help=
      "Path to results json file. Expects this file to have been pre-populated."
  )
  argParser.add_argument("-bid",
                         "--benchmark_id",
                         help="The unique id that defines a benchmark.")
  argParser.add_argument("-iter",
                         "--iterations",
                         type=int,
                         default=10,
                         help="The number of iterations to benchmark.")
  argParser.add_argument(
      "-d",
      "--device",
      default="gpu",
      help="The device to run on. Currently `cpu` and `gpu` are supported.")
  argParser.add_argument("--hlo_benchmark_path",
                         default=None,
                         help="The path to `run_hlo_module`.")

  args = argParser.parse_args()

  model_definition = benchmark_lookup(args.benchmark_id)
  print(f"\n\n--- {args.benchmark_id} -------------------------------------")

  benchmark_definition = {
      "benchmark_id": args.benchmark_id,
      "benchmark_name": model_definition.name,
      "framework": str(model_definition.meta_model.framework_type),
      "data_type": str(model_definition.meta_model.data_type),
      "batch_size": model_definition.input_batch_size,
      "inputs": model_definition.inputs.tensor_dimensions,
      "outputs": model_definition.outputs.tensor_dimensions,
      "compiler": "xla",
      "device": args.device,
      "tags": model_definition.meta_model.tags + model_definition.tags,
  }

  # Retrieve HLO input.
  if model_definition.meta_model.framework_type == data_types.ModelFrameworkType.JAX:
    hlo_artifact = model_definition.get_artifact(
        data_types.ModelArtifactType.JAX_HLO_DUMP)
  elif model_definition.meta_model.framework_type == data_types.ModelFrameworkType.TENSORFLOW_V2:
    hlo_artifact = model_definition.get_artifact(
        data_types.ModelArtifactType.TF_HLO_DUMP)

  local_hlo_path = "/tmp/hlo_input.txt"
  r = requests.get(hlo_artifact.source_url)
  open(local_hlo_path, 'wb').write(r.content)

  # Retrieve compiler-level benchmarks.
  compiler_metrics = run_compiler_benchmark(args.hlo_benchmark_path,
                                            local_hlo_path, args.iterations,
                                            args.device)

  result = {
      "definition": benchmark_definition,
      "metrics": {
          "compiler_level": compiler_metrics,
      },
  }
  print(json.dumps(result, indent=2))
  dump_result(args.output_path, result)
