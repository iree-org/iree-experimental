import argparse
import json
import os
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

TIME_UNITS = {"us": 1e-3, "ms": 1, "s": 1e3, "min": 60 * 1e3, "h": 3600 * 1e3}
TIME_REGEXP = re.compile(r"time: (\d+\.?\d*) (%s)" % "|".join(TIME_UNITS))
SIZE_REGEXP = re.compile(r" (\d+) bytes")


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


def parse_log_time(line: bytes) -> float:
  """Parses timestamp from the standard log.
  """
  match = re.search(rb"^(\d{4}-\d{2}-\d{2}) (\d{2}):(\d{2}):(\d{2}\.\d+):",
                    line)
  assert match, "Unable to parse log time: %s" % line
  _, h, m, s = match.groups()
  return 1000 * (int(h) * 3600 + int(m) * 60 + float(s))


def parse_log_elapsed_time(line1: bytes, line2: bytes) -> float:
  """Calculates elapsed time between two log lines.
  """
  start, end = parse_log_time(line1), parse_log_time(line2)
  end += 86400 if end < start else 0  # next day correction
  return end - start


def parse_latencies(raw_output: bytes, expected_iterations: int) -> list[float]:
  """Returns a list of latencies in milliseconds parsed from XLA logs.
  """
  start_regex = re.compile(rb".+HloRunner: ExecuteOnDevices started")
  start_matches = re.findall(start_regex, raw_output)

  stop_regex = re.compile(rb".+HloRunner: ExecuteOnDevices succeeded")
  stop_matches = re.findall(stop_regex, raw_output)

  assert len(start_matches) == len(
      stop_matches) == expected_iterations, "Unable to parse output."
  latencies = [
      parse_log_elapsed_time(t1, t2)
      for t1, t2 in zip(start_matches, stop_matches)
  ]
  return latencies


def parse_log_duration(time_str: bytes) -> float:
  """Returns the time in milliseconds parsed from XLA logs.
  """
  match = TIME_REGEXP.search(time_str.decode())
  assert match, "Unable to parse the time on log line"
  exp = TIME_UNITS[match.group(2)]
  return float(match.group(1)) * exp


def parse_log_size(size_str: bytes) -> float:
  """Returns the size in bytes parsed from XLA logs.
  """
  match = SIZE_REGEXP.search(size_str.decode())
  assert match, "Unable to parse the size on log line"
  return float(match.group(1)) * 1e-6


def parse_compile_time(raw_output: bytes,
                       expected_iterations: int) -> list[float]:
  compile_regex = re.compile(
      rb"NVPTXCompiler::CompileTargetBinary - CompileToPtx.*")
  matches = re.findall(compile_regex, raw_output)
  total_compile_time_ms = sum([parse_log_duration(t1) for t1 in matches])
  return total_compile_time_ms * 1e-3


def parse_peak_memory(raw_output: bytes) -> float:
  regex = re.compile(rb"New Peak memory usage of \d+ bytes for GPU")
  matches = re.findall(regex, raw_output)
  assert matches, "Unable to find peak memory"
  return parse_log_size(matches[-1])


def run_compiler_benchmark_gpu(hlo_benchmark_tool_path: str,
                               hlo_input_path: str, benchmark_iterations: int,
                               device: str) -> dict:
  assert hlo_benchmark_tool_path.endswith("hlo_runner_main")

  cmd = [
      hlo_benchmark_tool_path,
      f"--hlo_file={hlo_input_path}",
      f"--device_type={device}",
      f"--num_repeats={benchmark_iterations}",
      "--input_format=text",
      "--num_replicas=1",
      "--num_partitions=1",
      "--logtostderr",
  ]

  # Timings are logged under VLOG so we need to enable this.
  os.environ["TF_CPP_MIN_VLOG_LEVEL"] = "0"
  os.environ["TF_CPP_MAX_VLOG_LEVEL"] = "2"

  result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  result_text = result.stdout

  latencies = parse_latencies(result_text, benchmark_iterations)
  compile_time_s = parse_compile_time(result_text, benchmark_iterations)
  peak_memory_usage = parse_peak_memory(result_text)

  results_dict = {
      "compile_time_s": compile_time_s,
      "min_latency_ms": min(latencies, default=None),
      "max_latency_ms": max(latencies, default=None),
      "mean_latency_ms": statistics.mean(latencies) if latencies else None,
      "median_latency_ms": statistics.median(latencies) if latencies else None,
      "stddev_latency_ms": statistics.stdev(latencies) if latencies else None,
      "benchmark_iterations": benchmark_iterations,
      "device_memory_peak_mb": peak_memory_usage,
  }
  return results_dict


def run_compiler_benchmark_cpu(hlo_benchmark_tool_path: str,
                               hlo_input_path: str, benchmark_iterations: int,
                               device: str) -> dict:
  assert hlo_benchmark_tool_path.endswith("run_hlo_module")

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
      required=True,
      help=
      "Path to results json file. Expects this file to have been pre-populated."
  )
  argParser.add_argument("-bid",
                         "--benchmark_id",
                         required=True,
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
                         required=True,
                         help="The path to `run_hlo_module`.")
  argParser.add_argument("--cache_dir",
                         required=True,
                         help="The path to save HLO artifacts")

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

  assert hlo_artifact.source_url.startswith("https://storage.googleapis.com/iree-model-artifacts/")
  relative_path = hlo_artifact.source_url.removeprefix("https://storage.googleapis.com/iree-model-artifacts/")
  hlo_local_path = os.path.join(args.cache_dir, relative_path)

  if not os.path.exists(hlo_local_path):
    pathlib.Path(os.path.dirname(hlo_local_path)).mkdir(parents=True, exist_ok=True)
    r = requests.get(hlo_artifact.source_url)
    print(f"Downloading {hlo_artifact.source_url}")
    open(hlo_local_path, 'wb').write(r.content)
  else:
    print(f"{hlo_artifact.source_url} already downloaded.")

  # Retrieve compiler-level benchmarks.
  # We use different binaries for benchmarking gpu and cpu.
  if args.device == "gpu":
    compiler_metrics = run_compiler_benchmark_gpu(args.hlo_benchmark_path,
                                                  hlo_local_path,
                                                  args.iterations, args.device)
  else:
    compiler_metrics = run_compiler_benchmark_cpu(args.hlo_benchmark_path,
                                                  hlo_local_path,
                                                  args.iterations, args.device)


  result = {
      "definition": benchmark_definition,
      "metrics": {
          "compiler_level": compiler_metrics,
      },
  }
  print(json.dumps(result, indent=2))
  dump_result(args.output_path, result)
