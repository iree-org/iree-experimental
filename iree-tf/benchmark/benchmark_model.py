import argparse
import json
import multiprocessing
import numpy as np
import os
import pathlib
import shutil
import statistics
import sys
import tensorflow as tf
import time

from typing import Optional

# Add library dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "library"))
from models import resnet50, bert_large, t5_large

# Add benchmark definitions to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "oobi" / "benchmark-definitions" / "python"))
import tf_model_definitions, unique_ids, utils


_HLO_DUMP_DIR = "/tmp/hlo_dump"
_TF_CPU_DEVICE = "/CPU:0"
_TF_GPU_DEVICE = "/GPU:0"


def benchmark_lookup(unique_id: str):
  if unique_id not in tf_model_definitions.TF_MODELS_DICT:
    id_list = '\n  '.join(tf_model_definitions.TF_MODELS_DICT.keys())
    raise ValueError(f"Id {unique_id} does not exist in model suite. Expected "
                     f"one of:\n  {id_list}")

  model_definition = tf_model_definitions.TF_MODELS_DICT[unique_id]
  if unique_id.startswith(unique_ids.MODEL_RESNET50_FP32_TF):
    return ("RESNET50", resnet50.ResNet50, model_definition)
  elif unique_id.startswith(unique_ids.MODEL_BERT_LARGE_FP32_TF):
    return ("BERT_LARGE", bert_large.BertLarge, model_definition)
  elif unique_id.startswith(unique_ids.MODEL_T5_LARGE_FP32_TF):
    return ("T5_LARGE", t5_large.T5Large, model_definition)
  else:
    raise ValueError(f"Model definition not supported")


def dump_result(file_path: str, result: dict) -> None:
  with open(file_path, "r") as f:
    dictObj = json.load(f)

  dictObj["execution_environment"] = {
      "python_environment": utils.get_python_environment_info()
  }
  dictObj["benchmarks"].append(result)

  with open(file_path, "w") as f:
    json.dump(dictObj, f)


def bytes_to_mb(bytes: Optional[int]) -> Optional[float]:
  return None if bytes is None else bytes / 1e6


def run_framework_benchmark(model_name: str, model_class: type[tf.Module],
                            input_data: tuple[np.array, ...],
                            expected_outputs: tuple[np.array, ...],
                            warmup_iterations: int, benchmark_iterations: int,
                            tf_device: str, shared_dict) -> None:
  try:
    with tf.device(tf_device):
      if tf_device == _TF_GPU_DEVICE:
        tf.config.experimental.reset_memory_stats(tf_device)

      model = model_class()

      # Run warmup.
      warmup_latencies = []
      for i in range(warmup_iterations):
        start = time.perf_counter()
        outputs = model.forward(*input_data)
        tf.test.experimental.sync_devices()
        latency = 1000 * (time.perf_counter() - start)
        utils.compare_results(outputs, expected_outputs[0])
        warmup_latencies.append(latency)

      # Run benchmark.
      latencies = []
      for i in range(benchmark_iterations):
        start = time.perf_counter()
        outputs = model.forward(*input_data)
        tf.test.experimental.sync_devices()
        latency = 1000 * (time.perf_counter() - start)
        utils.compare_results(outputs, expected_outputs[0])
        latencies.append(latency)

      # Retrieve memory stats.
      if tf_device == _TF_GPU_DEVICE:
        memory_info = tf.config.experimental.get_memory_info(tf_device)
        device_peak_b = memory_info["peak"]
      else:
        # tf.config.experimental does not currently support measuring CPU memory usage.
        device_peak_b = None
      device_peak_mb = bytes_to_mb(device_peak_b)

      compile_time_s = None if not warmup_latencies else (
          max(warmup_latencies) - statistics.median(latencies)) / 1000

      # Save results.
      result_dict = {
          "min_warmup_latency_ms": min(warmup_latencies, default=None),
          "max_warmup_latency_ms": max(warmup_latencies, default=None),
          "mean_warmup_latency_ms": None if not warmup_latencies else statistics.mean(warmup_latencies),
          "median_warmup_latency_ms": None if not warmup_latencies else statistics.median(warmup_latencies),
          "stddev_warmup_latency_ms": None if not warmup_latencies else statistics.stdev(warmup_latencies),
          "warmup_iterations": warmup_iterations,
          "min_latency_ms": min(latencies, default=None),
          "max_latency_ms": max(latencies, default=None),
          "mean_latency_ms": None if not latencies else statistics.mean(latencies),
          "median_latency_ms": None if not latencies else statistics.median(latencies),
          "stddev_latency_ms": None if not latencies else statistics.stdev(latencies),
          "benchmark_iterations": benchmark_iterations,
          "compile_time_s": compile_time_s,
          "device_memory_peak_mb": device_peak_mb,
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
  argParser.add_argument(
      "--run_in_process",
      action="store_true",
      help="Whether to run the benchmark under the same process. Set this to true when profiling a single workload")
  argParser.add_argument("--cache_dir",
                         required=True,
                         type=pathlib.Path,
                         help="Directory to download artifacts to.")

  args = argParser.parse_args()

  model_name, model_class, model_definition = benchmark_lookup(
      args.benchmark_id)
  print(
      f"\n\n--- {model_name} {args.benchmark_id} -------------------------------------"
  )

  if os.path.exists(_HLO_DUMP_DIR):
    shutil.rmtree(_HLO_DUMP_DIR)
  os.mkdir(_HLO_DUMP_DIR)

  batch_size = model_definition.input_batch_size
  benchmark_definition = {
      "benchmark_id": args.benchmark_id,
      "benchmark_name": model_definition.name,
      "framework": str(model_definition.meta_model.framework_type),
      "data_type": str(model_definition.meta_model.data_type),
      "batch_size": batch_size,
      "inputs": model_definition.inputs.tensor_dimensions,
      "outputs": model_definition.outputs.tensor_dimensions,
      "compiler": "xla",
      "device": args.device,
      "tags": model_definition.meta_model.tags + model_definition.tags,
  }

  inputs = utils.retrieve_model_data(model_definition.inputs, args.cache_dir)
  expected_outputs = utils.retrieve_model_data(model_definition.outputs,
                                               args.cache_dir)

  framework_metrics = {}
  tf_device = _TF_GPU_DEVICE if args.device == "gpu" else _TF_CPU_DEVICE

  with multiprocessing.Manager() as manager:
    shared_dict = manager.dict()

    if args.run_in_process:
      run_framework_benchmark(model_name, model_class, inputs, expected_outputs,
                              args.warmup_iterations, args.iterations,
                              tf_device, shared_dict)
    else:
      p = multiprocessing.Process(target=run_framework_benchmark,
                                  args=(model_name, model_class, inputs,
                                        expected_outputs,
                                        args.warmup_iterations, args.iterations,
                                        tf_device, shared_dict))
      p.start()
      p.join()

    framework_metrics.update(shared_dict)

  result = {
      "definition": benchmark_definition,
      "metrics": {
          "framework_level": framework_metrics,
      },
  }
  print(json.dumps(result, indent=2))
  dump_result(args.output_path, result)
