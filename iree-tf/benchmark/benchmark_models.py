import argparse
import glob
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

from typing import Optional, Dict

# Add library dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "library"))
from models import resnet50, bert_large, t5_large

_MODEL_NAME_TO_MODEL_CONFIG = {
    # Batch sizes taken from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/resnet50
    "RESNET50": (resnet50.ResNet50, [1, 8, 64, 128, 256, 2048]),
    # Batch sizes based on MLPerf config: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/bert
    "BERT_LARGE":
    (bert_large.BertLarge, [1, 16, 24, 32, 48, 64, 512, 1024, 1280]),
    # Uses the same batch sizes as Bert-Large
    "T5_LARGE": (t5_large.T5Large, [1, 16, 24, 32, 48, 64, 512]),
}

_HLO_DUMP_DIR = "/tmp/hlo_dump"
_TF_CPU_DEVICE = "/CPU:0"
_TF_GPU_DEVICE = "/GPU:0"

def write_line(file_path: str, text: str, append: bool = True) -> None:
    with open(file_path, "a" if append else "w") as f:
        print(f"Writing {text}")
        f.write(text + "\n")


def bytes_to_mb_str(bytes: Optional[int]) -> str:
    return "n/a" if bytes is None else f"{bytes / 1e6:.6f}"


def run_framework_benchmark(model_name: str, model_class: type[tf.Module],
                            batch_size: int, warmup_iterations: int,
                            benchmark_iterations: int, tf_device: str,
                            hlo_dump_dir: str, shared_dict) -> None:
    try:
        with tf.device(tf_device):
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
                latency = 1000 * (time.perf_counter() - start)
                warmup_latencies.append(latency)

            # Run benchmark.
            latencies = []
            for i in range(benchmark_iterations):
                start = time.perf_counter()
                model.forward(*inputs)
                latency = 1000 * (time.perf_counter() - start)
                latencies.append(latency)

            # Retrieve memory stats.
            if tf_device == _TF_GPU_DEVICE:
                memory_info = tf.config.experimental.get_memory_info(tf_device)
                device_peak_b = memory_info["peak"]
            else:
                # tf.config.experimental does not currently support measuring CPU memory usage.
                device_peak_b = None
            device_peak_mb = bytes_to_mb_str(device_peak_b)

            compile_time_s = (max(warmup_latencies) -
                              statistics.median(latencies)) / 1000

            # Save results.
            result_dict = {
                "framework_min_warmup_latency_ms":
                str(min(warmup_latencies)),
                "framework_max_warmup_latency_ms":
                str(max(warmup_latencies)),
                "framework_mean_warmup_latency_ms":
                str(statistics.mean(warmup_latencies)),
                "framework_median_warmup_latency_ms":
                str(statistics.median(warmup_latencies)),
                "framework_stddev_warmup_latency_ms":
                str(statistics.stdev(warmup_latencies)),
                "framework_warmup_iterations":
                str(warmup_iterations),
                "framework_min_latency_ms":
                str(min(latencies)),
                "framework_max_latency_ms":
                str(max(latencies)),
                "framework_mean_latency_ms":
                str(statistics.mean(latencies)),
                "framework_median_latency_ms":
                str(statistics.median(latencies)),
                "framework_stddev_latency_ms":
                str(statistics.stdev(latencies)),
                "framework_benchmark_iterations":
                str(benchmark_iterations),
                "framework_compile_time_s":
                str(compile_time_s),
                "device_memory_peak_mb":
                str(device_peak_mb),
            }
            shared_dict.update(result_dict)

    except Exception as e:
        print(f"Failed to benchmark model {model_name}. Exception: {e}")


def run_compiler_benchmark(hlo_benchmark_tool_path: str, hlo_dir: str,
                           device: str, shared_dict) -> None:
    hlo_file = glob.glob(f"{hlo_dir}/*.before_optimizations.txt")
    assert (len(hlo_file) == 1)

    cmd = [
        hlo_benchmark_tool_path, "--input_format=hlo", f"--platform={device}",
        "--reference_platform=", "--logtostderr",
        f"--input_module={hlo_file[0]}"
    ]
    result = subprocess.run(cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    result_text = result.stdout.decode("utf-8")
    print(f"result: {result_text}")

    regex = re.compile(r"... compiled and ran in (.*)s.")
    matches = re.findall(regex, result_text)
    assert (len(matches) == 1)
    compiler_compile_time_s = matches[0]
    print(f"compiler_compile_time_s: {compiler_compile_time_s}")

    regex = re.compile(r"execution time for runner CUDA: (.*)s.")
    matches = re.findall(regex, result_text)
    compiler_latency_ms = float(matches[0]) * 1000

    shared_dict.update({
        "compiler_compile_time_s": str(compiler_compile_time_s),
        "compiler_latency_ms": str(compiler_latency_ms),
    })


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-o",
                           "--output_path",
                           default="/tmp/tf_benchmarks.csv",
                           help="Path to results csv file.")
    argParser.add_argument("-w",
                           "--warmup_iterations",
                           default=5,
                           help="The number of warmup steps.")
    argParser.add_argument("-iter",
                           "--iterations",
                           default=100,
                           help="The number of iterations to benchmark.")
    argParser.add_argument("-d",
                           "--device",
                           default="gpu",
                           help="The device to run on. Currently `cpu` and `gpu` are supported.")
    argParser.add_argument("--hlo_benchmark_path",
                           help="The path to `run_hlo_module`.")
    args = argParser.parse_args()
    tf_device = _TF_GPU_DEVICE if args.device == "gpu" else _TF_CPU_DEVICE

    header_written = False
    for model_name, model_config in _MODEL_NAME_TO_MODEL_CONFIG.items():
        print(f"\n\n--- {model_name} -------------------------------------")
        model_class, batch_sizes = model_config

        for batch_size in batch_sizes:
            if os.path.exists(_HLO_DUMP_DIR):
                shutil.rmtree(_HLO_DUMP_DIR)
            os.mkdir(_HLO_DUMP_DIR)

            result_dict = {
                "model": model_name,
                "batch_size": str(batch_size),
            }

            # Retrieve framework-level benchmarks.
            with multiprocessing.Manager() as manager:
                shared_dict = manager.dict()
                p = multiprocessing.Process(
                    target=run_framework_benchmark,
                    args=(model_name, model_class, batch_size,
                          args.warmup_iterations, args.iterations, tf_device,
                          _HLO_DUMP_DIR, shared_dict))
                p.start()
                p.join()
                result_dict.update(shared_dict)

            # Retrieve compiler-level benchmarks.
            with multiprocessing.Manager() as manager:
                shared_dict = manager.dict()
                p = multiprocessing.Process(
                    target=run_compiler_benchmark,
                    args=(args.hlo_benchmark_path, _HLO_DUMP_DIR,
                          "cuda" if args.device == "gpu" else "cpu",
                          shared_dict))
                p.start()
                p.join()
                result_dict.update(shared_dict)

            if not header_written:
                write_line(args.output_path,
                           ",".join(result_dict.keys()),
                           append=False)
                header_written = True
            write_line(args.output_path,
                       ",".join(result_dict.values()),
                       append=True)
