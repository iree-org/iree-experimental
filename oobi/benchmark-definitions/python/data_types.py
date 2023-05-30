import dataclasses
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

import serialization


class ModelFrameworkType(Enum):
  """Type of framework a model is implemented in."""
  TENSORFLOW_V1 = "tensorflow_v1"
  TENSORFLOW_V2 = "tensorflow_v2"
  PYTORCH = "framework_pt"
  JAX = "framework_jax"


class ModelArtifactType(Enum):
  """Type of model artifact."""
  TF_SAVEDMODEL_V1 = "tf_savedmodel_v1"
  TF_SAVEDMODEL_V2 = "tf_savedmodel_v2"
  TF_HLO_DUMP = "tf_hlo_dump"
  JAX_HLO_DUMP = "jax_hlo_dump"
  MLIR_STABLEHLO = "mlir_stablehlo"
  MLIR_MHLO = "mlir_mhlo"
  MLIR_LINALG = "mlir_linalg"
  MLIR_TOSA = "mlir_tosa"


class DataType(Enum):
  """Model data type used in the model."""
  FP32 = "fp32"
  FP16 = "fp16"
  BF16 = "bf16"
  INT8 = "int8"
  UINT8 = "uint8"

class DataFormat(Enum):
  """Model input data format."""
  ZEROS = "zeros"
  NUMPY_NPY = "numpy_npy"


@serialization.serializable
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


@serialization.serializable
@dataclass(frozen=True)
class ModelArtifact(object):
  """An artifact derived from a model"""
  artifact_type: ModelArtifactType
  # Where to download the model artifact.
  source_url: str


@serialization.serializable
@dataclass(frozen=True)
class MetaModel(object):
  """ A model implementation without concrete inputs."""
  id: str
  name: str
  tags: List[str]
  framework_type: ModelFrameworkType
  # Source of the model implementation.
  source_info: str
  data_type: DataType


@serialization.serializable
@dataclass(frozen=True)
class Model(object):
  """A Model implementation"""
  id: str
  # Friendly unique name.
  name: str
  # Tags that describe the model characteristics.
  tags: List[str]
  meta_model: MetaModel
  input_batch_size: int
  inputs: ModelData
  outputs: ModelData
  # A list of artifacts derived from this model.
  artifacts: List[ModelArtifact]

  def __str__(self):
    return self.name

  def get_artifact(self, artifact_type: ModelFrameworkType) -> ModelArtifact:
    for artifact in self.artifacts:
      if artifact.artifact_type == artifact_type:
        return artifact
    return None
