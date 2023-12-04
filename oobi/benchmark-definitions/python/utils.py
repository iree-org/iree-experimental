import numpy as np
import os
import pathlib
import re
import requests
import subprocess

from typing import Any

import data_types


def get_python_environment_info():
  """ Returns a dictionary of package versions in the python virtual environment."""
  output = subprocess.check_output(["pip", "list"]).decode("utf-8")
  # The first few lines are the table headers so we remove that.
  output = output[output.rindex("---\n") + 4:]
  output = output.split("\n")
  package_dict = {}
  for item in output:
    split = re.split("\s+", item)
    if len(split) == 2:
      package_dict[split[0]] = split[1]
  return package_dict


def download_file(source_url: str, save_path: str):
  """ Downloads `source_url` to `saved_path`."""
  save_path.parent.mkdir(parents=True, exist_ok=True)
  with requests.get(source_url, stream=True) as response:
    with open(save_path, "wb") as f:
      for chunk in response.iter_content(chunk_size=1024):
        f.write(chunk)
  print(f"Downloaded {source_url} to {save_path}")


def retrieve_model_data(
    model_data: data_types.ModelData,
    cache_dir: str,
    url_prefix: str = "https://storage.googleapis.com/iree-model-artifacts/"
) -> tuple[Any, ...]:
  """ Downloads all artifacts listed in `model_data` into `cache_dir`.

  Replicates relative path of source URL into `cache_dir`.
  """
  data = ()
  for url in model_data.source_url:
    assert url.startswith(url_prefix)
    relative_path = url.removeprefix(url_prefix)
    local_path = cache_dir / relative_path

    if not local_path.exists():
      download_file(url, local_path)

    array = np.load(local_path)
    data = data + (array,)
  return data


def compare_results(a: np.array, b: np.array, atol=0.5):
  is_equal = np.allclose(a, b, atol=atol)
  if not is_equal:
    max_diff = np.max(np.abs(b - a))
    raise RuntimeError(f"Outputs do not match expected. Max diff: {max_diff}")
