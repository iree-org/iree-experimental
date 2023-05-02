from dataclasses import dataclass
from enum import Enum
from typing import List

class FrameworkType(Enum):
  """Type of model source."""
  TENSORFLOW_V1 = "tensorflow_v1"
  TENSORFLOW_V2 = "tensorflow_v2"
  PYTORCH = "framework_pt"

class DataFormat(Enum):
  """Model input data format."""
  ZEROS = "zeros"
  NUMPY_NPY = "numpy_npy"
   
@dataclass(frozen=True)
class ModelData(object):
  """Input or output data to benchmark the model."""
  id: str
  # Friendly name.
  name: str
  # Tags that describe the data characteristics.
  tags: List[str]
  data_format: DataFormat
  # If applicable, the model id that generated the data.
  model_id: str
  # Information on where the data was originally sourced.
  source_info: str
  # The name of the tensors.
  tensor_names: List[str]
  # The dimensions of the data e.g. "1x224x224x3xf32".
  tensor_dimensions: List[str]
  # Where to download the input data.
  source_url: List[str]

  def __str__(self):
    return self.name

@dataclass(frozen=True)
class Model(object):
  """A Model implementation"""
  id: str
  # Friendly unique name.
  name: str
  # Tags that describe the model characteristics.
  tags: List[str]
  framework_type: FrameworkType
  # Source of the model implementation.
  source_info: str
  input_batch_size: int
  inputs: ModelData
  outputs: ModelData

  def __str__(self):
    return self.name
