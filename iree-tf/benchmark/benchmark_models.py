import argparse
import pathlib
import statistics
import sys
import time

from multiprocessing import Process

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


def write_line(file_path, text, append=True):
    with open(file_path, "a" if append else "w") as f:
        print(f"Writing {text}")
        f.write(text + "\n")


def run_benchmark(model_name, model_class, batch_size, warmup_iterations,
                  benchmark_iterations, csv_path):
    try:
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

        # Save results.
        result = f"{model_name},{batch_size},{min(warmup_latencies)},{max(warmup_latencies)},{statistics.mean(warmup_latencies)},{statistics.median(warmup_latencies)},{statistics.stdev(warmup_latencies)},{min(latencies)},{max(latencies)},{statistics.mean(latencies)},{statistics.median(latencies)},{statistics.stdev(latencies)}"
        write_line(csv_path, result)

    except Exception as e:
        print(f"Failed to benchmark model {model_name}. Exception: {e}")


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
    args = argParser.parse_args()

    csv_header = "model,batch_size,warmup_min_ms,warmup_max_ms,warmup_mean_ms,warmup_median_ms,warmup_stddev_ms,latency_min_ms,latency_max_ms,latency_mean_ms,latency_median_ms,latency_stddev_ms"
    write_line(args.output_path, csv_header, append=False)

    for model_name, model_config in _MODEL_NAME_TO_MODEL_CONFIG.items():
        print(f"\n\n--- {model_name} -------------------------------------")
        model_class, batch_sizes = model_config

        for batch_size in batch_sizes:
            p = Process(target=run_benchmark,
                        args=(model_name, model_class, batch_size,
                              args.warmup_iterations, args.iterations,
                              args.output_path))
            p.start()
            p.join()
