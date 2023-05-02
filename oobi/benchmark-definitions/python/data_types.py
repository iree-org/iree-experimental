from dataclasses import dataclass
from enum import Enum
from typing import List

class ModelFrameworkType(Enum):
  """Type of framework a model is implemented in."""
  TENSORFLOW_V1 = "tensorflow_v1"
  TENSORFLOW_V2 = "tensorflow_v2"
  PYTORCH = "framework_pt"

class ModelArtifactType(Enum):
  """Type of model artifact."""
  TF_SAVEDMODEL_V1 = "tf_savedmodel_v1"
  TF_SAVEDMODEL_V2 = "tf_savedmodel_v2"
  MLIR_STABLEHLO = "mlir_stablehlo"
  MLIR_MHLO = "mlir_mhlo"
  MLIR_LINALG = "mlir_linalg"
  MLIR_TOSA = "mlir_tosa"

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
  framework_type: ModelFrameworkType
  # Source of the model implementation.
  source_info: str
  input_batch_size: int
  inputs: ModelData
  outputs: ModelData

  def __str__(self):
    return self.name
  
@dataclass(frozen=True)
class ModelArtifact(object):
  """An artifact derived from a model"""
  id: str
  # The id of the model implementation the artifact is derived from.
  parent_model_id: str
  # Friendly unique name.
  name: str
  # Tags that describe the artifact characteristics.
  tags: List[str]
  artifact_type: ModelArtifactType
  # Where to download the model artifact.
  source_url: str
