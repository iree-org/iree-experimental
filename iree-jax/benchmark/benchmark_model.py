import argparse
import jax
import json
import multiprocessing
import pathlib
import statistics
import sys
import time

from typing import Optional, Any


# Add library dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "library"))
from models import bert_large

# Add benchmark definitions to the search path.
sys.path.insert(
    0,
    str(
        pathlib.Path(__file__).parent.parent.parent / "oobi" /
        "benchmark-definitions" / "python"))
import data_types, jax_model_definitions, unique_ids


def benchmark_lookup(unique_id: str):
  if unique_id not in jax_model_definitions.JAX_MODELS_DICT:
    raise ValueError(f"Id {unique_id} does not exist in model suite.")

  model_definition = jax_model_definitions.JAX_MODELS_DICT[unique_id]
  if unique_id.startswith(unique_ids.MODEL_BERT_LARGE_FP32_JAX):
    return ("BERT_LARGE", bert_large.BertLarge, model_definition)
  else:
    raise ValueError(f"Model definition not supported")


def dump_result(file_path: str, result: dict) -> None:
  with open(file_path, "r") as f:
    dictObj = json.load(f)

  dictObj["benchmarks"].append(result)
  with open(file_path, "w") as f:
    json.dump(dictObj, f)


def bytes_to_mb_str(bytes: Optional[int]) -> str:
  return "n/a" if bytes is None else f"{bytes / 1e6:.6f}"


def run_framework_benchmark(model_name: str, model_class: Any, batch_size: int,
                            warmup_iterations: int, benchmark_iterations: int,
                            backend: str, shared_dict) -> None:
  try:
    with jax.default_device(jax.devices(backend)[0]):
      model = model_class()
      inputs = model.generate_inputs(batch_size)

      # Create jits.
      jit_inputs = jax.device_put(inputs)
      jit_function = jax.jit(model.forward)

      # Run warmup.
      warmup_latencies = []
      for i in range(warmup_iterations):
        start = time.perf_counter()
        jax.block_until_ready(jit_function(*jit_inputs))
        latency = 1000 * (time.perf_counter() - start)
        warmup_latencies.append(latency)

      # Run benchmark.
      latencies = []
      for i in range(benchmark_iterations):
        start = time.perf_counter()
        jax.block_until_ready(jit_function(*jit_inputs))
        latency = 1000 * (time.perf_counter() - start)
        latencies.append(latency)

      # Save results.
      result_dict = {
        "min_warmup_latency_ms":
           "n/a" if not warmup_latencies else str(min(warmup_latencies)),
        "max_warmup_latency_ms":
            "n/a" if not warmup_latencies else str(max(warmup_latencies)),
        "mean_warmup_latency_ms":
            "n/a" if not warmup_latencies else str(
                statistics.mean(warmup_latencies)),
        "median_warmup_latency_ms":
            "n/a" if not warmup_latencies else str(
                statistics.median(warmup_latencies)),
        "stddev_warmup_latency_ms":
            "n/a" if not warmup_latencies else str(
                statistics.stdev(warmup_latencies)),
        "warmup_iterations":
            str(warmup_iterations),
        "min_latency_ms":
            str(min(latencies)),
        "max_latency_ms":
            str(max(latencies)),
        "mean_latency_ms":
            str(statistics.mean(latencies)),
        "median_latency_ms":
            str(statistics.median(latencies)),
        "stddev_latency_ms":
            str(statistics.stdev(latencies)),
        "benchmark_iterations":
            str(benchmark_iterations),
      }
      shared_dict.update(result_dict)

  except Exception as e:
    print(f"Failed to benchmark model {model_name}. Exception: {e}")


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
  argParser.add_argument("-w",
                         "--warmup_iterations",
                         type=int,
                         default=5,
                         help="The number of warmup steps.")
  argParser.add_argument("-iter",
                         "--iterations",
                         type=int,
                         default=100,
                         help="The number of iterations to benchmark.")
  argParser.add_argument(
      "-d",
      "--device",
      default="gpu",
      help="The device to run on. Currently `cpu` and `gpu` are supported.")
  argParser.add_argument("--hlo_benchmark_path",
                         default=None,
                         help="The path to `run_hlo_module`.")
  argParser.add_argument(
      "--run_in_process",
      action=argparse.BooleanOptionalAction,
      help=
      "Whether to run the benchmark under the same process. Set this to true when profiling a single workload"
  )

  args = argParser.parse_args()

  model_name, model_class, model_definition = benchmark_lookup(
      args.benchmark_id)
  print(
      f"\n\n--- {model_name} {args.benchmark_id} -------------------------------------"
  )

  batch_size = model_definition.input_batch_size
  benchmark_definition = {
      "benchmark_id": args.benchmark_id,
      "benchmark_name": model_definition.name,
      "batch_size": str(batch_size),
      "framework": "jax",
      "compiler": "xla",
      "device": args.device,
  }

  framework_metrics = {}
  # Retrieve framework-level benchmarks.
  with multiprocessing.Manager() as manager:
    shared_dict = manager.dict()

    if args.run_in_process:
      run_framework_benchmark(model_name, model_class, batch_size,
                              args.warmup_iterations, args.iterations,
                              args.device, shared_dict)
    else:
      p = multiprocessing.Process(target=run_framework_benchmark,
                                  args=(model_name, model_class, batch_size,
                                        args.warmup_iterations, args.iterations,
                                        args.device, shared_dict))
      p.start()
      p.join()

    framework_metrics.update(shared_dict)

  result = {
      "definition": benchmark_definition,
      "metrics": {
          "framework_level": framework_metrics,
      }
  }
  print(result)
  dump_result(args.output_path, result)
