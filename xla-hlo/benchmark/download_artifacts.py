import argparse
import pathlib
import sys

from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import wait

# Add benchmark definitions to the search path.
sys.path.insert(
    0,
    str(
        pathlib.Path(__file__).parent.parent.parent / "oobi" /
        "benchmark-definitions" / "python"))
import data_types, model_dictionary, utils


if __name__ == "__main__":
  argParser = argparse.ArgumentParser()
  argParser.add_argument("-o",
                         "--output_dir",
                         required=True,
                         type=pathlib.Path,
                         help="Directory to download artifacts to.")
  argParser.add_argument(
      "-bids",
      "--benchmark_ids",
      nargs="+",
      required=True,
      help="A list of benchmark ids to download artifacts for.")
  args = argParser.parse_args()

  artifacts = []
  for bid in args.benchmark_ids:
    model_definition = model_dictionary.get_model_definition(bid)
    # Retrieve HLO input.
    if model_definition.meta_model.framework_type == data_types.ModelFrameworkType.JAX:
      hlo_artifact = model_definition.get_artifact(
          data_types.ModelArtifactType.JAX_HLO_DUMP)
    elif model_definition.meta_model.framework_type == data_types.ModelFrameworkType.TENSORFLOW_V2:
      hlo_artifact = model_definition.get_artifact(
          data_types.ModelArtifactType.TF_HLO_DUMP)

    assert hlo_artifact.source_url.startswith(
        "https://storage.googleapis.com/iree-model-artifacts/")
    relative_path = hlo_artifact.source_url.removeprefix(
        "https://storage.googleapis.com/iree-model-artifacts/")
    hlo_local_path = args.output_dir / relative_path

    artifacts.append((hlo_artifact.source_url, hlo_local_path))

  with ProcessPoolExecutor(8) as exe:
    futures = [
        exe.submit(utils.download_file, source_url, local_path)
        for source_url, local_path in artifacts
    ]
    wait(futures)

  print("All downloads complete.")
