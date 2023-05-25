import argparse
import glob
import json
import multiprocessing
import os
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import tensorflow as tf
import time

from typing import Optional

# Add library dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "library"))
from models import resnet50, bert_large, t5_large

# Add benchmark definitions to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "oobi" / "benchmark-definitions" / "python"))
import tf_model_definitions, unique_ids
from utils import execution_environment


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
      "python_environment": execution_environment.get_python_environment_info()
  }
  dictObj["benchmarks"].append(result)

  with open(file_path, "w") as f:
    json.dump(dictObj, f)


def bytes_to_mb(bytes: Optional[int]) -> Optional[float]:
  return None if bytes is None else bytes / 1e6


def run_framework_benchmark(model_name: str, model_class: type[tf.Module],
                            batch_size: int, warmup_iterations: int,
                            benchmark_iterations: int, tf_device: str,
                            hlo_dump_dir: str, dump_hlo: bool, shared_dict) -> None:
  try:
    with tf.device(tf_device):
      if dump_hlo:
        # Configure to dump hlo.
        os.environ["XLA_FLAGS"] = f"--xla_dump_to={hlo_dump_dir}"

      if tf_device == _TF_GPU_DEVICE:
        tf.config.experimental.reset_memory_stats(tf_device)

      model = model_class()
      inputs = model.generate_inputs(batch_size)

      # Run warmup.
      warmup_latencies = []
      for i in range(warmup_iterations):
        start = time.perf_counter()
        model.forward(*inputs)
        tf.test.experimental.sync_devices()
        latency = 1000 * (time.perf_counter() - start)
        warmup_latencies.append(latency)

      # Run benchmark.
      latencies = []
      for i in range(benchmark_iterations):
        start = time.perf_counter()
        model.forward(*inputs)
        tf.test.experimental.sync_devices()
        latency = 1000 * (time.perf_counter() - start)
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


def run_compiler_benchmark(hlo_benchmark_tool_path: str, hlo_dir: str,
                           benchmark_iterations: int, device: str,
                           shared_dict) -> None:
  hlo_file = glob.glob(f"{hlo_dir}/*.before_optimizations.txt")
  assert (len(hlo_file) == 1)

  cmd = [
      hlo_benchmark_tool_path,
      "--input_format=hlo",
      f"--platform={device}",
      "--reference_platform=",
      "--logtostderr",
      f"--input_module={hlo_file[0]}",
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

  shared_dict.update({
      "compile_time_s": compile_time_latency,
      "min_latency_ms": min(latencies, default=None),
      "max_latency_ms": max(latencies, default=None),
      "mean_latency_ms": statistics.mean(latencies) if latencies else None,
      "median_latency_ms": statistics.median(latencies) if latencies else None,
      "stddev_latency_ms": statistics.stdev(latencies) if latencies else None,
      "benchmark_iterations": benchmark_iterations,
  })


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
      "--hlo_iterations",
      type=int,
      default=100,
      help="The number of iterations to run compiler-level benchmarks.")
  argParser.add_argument(
      "--run_in_process",
      action=argparse.BooleanOptionalAction,
      help="Whether to run the benchmark under the same process. Set this to true when profiling a single workload")

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

  framework_metrics = {}
  # Retrieve framework-level benchmarks.
  tf_device = _TF_GPU_DEVICE if args.device == "gpu" else _TF_CPU_DEVICE
  dump_hlo = False if args.hlo_benchmark_path is None else True
  with multiprocessing.Manager() as manager:
    shared_dict = manager.dict()

    if args.run_in_process:
      run_framework_benchmark(model_name, model_class, batch_size, args.warmup_iterations,
                              args.iterations, tf_device, _HLO_DUMP_DIR, dump_hlo,
                              shared_dict)
    else:
      p = multiprocessing.Process(target=run_framework_benchmark,
                                  args=(model_name, model_class, batch_size,
                                        args.warmup_iterations, args.iterations,
                                        tf_device, _HLO_DUMP_DIR, dump_hlo,
                                        shared_dict))
      p.start()
      p.join()

    framework_metrics.update(shared_dict)

  # Retrieve compiler-level benchmarks.
  compiler_metrics = {}
  if args.hlo_benchmark_path is not None:
    with multiprocessing.Manager() as manager:
      shared_dict = manager.dict()

      if args.run_in_process:
        run_compiler_benchmark(args.hlo_benchmark_path, _HLO_DUMP_DIR, args.hlo_iterations,
                              "cuda" if args.device == "gpu" else "cpu", shared_dict)
      else:
        p = multiprocessing.Process(
            target=run_compiler_benchmark,
            args=(args.hlo_benchmark_path, _HLO_DUMP_DIR, args.hlo_iterations,
                  "cuda" if args.device == "gpu" else "cpu", shared_dict))
        p.start()
        p.join()

      compiler_metrics.update(shared_dict)

  result = {
      "definition": benchmark_definition,
      "metrics": {
          "framework_level": framework_metrics,
          "compiler_level": compiler_metrics,
      },
  }
  print(json.dumps(result, indent=2))
  dump_result(args.output_path, result)
