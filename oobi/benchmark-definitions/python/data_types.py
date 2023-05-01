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


class ModelArtifactType(Enum):
  """Type of model artifact."""
  TF_SAVEDMODEL_V1 = "tf_savedmodel_v1"
  TF_SAVEDMODEL_V2 = "tf_savedmodel_v2"
  TF_HLO_DUMP = "tf_hlo_dump"
  MLIR_STABLEHLO = "mlir_stablehlo"
  MLIR_MHLO = "mlir_mhlo"
  MLIR_LINALG = "mlir_linalg"
  MLIR_TOSA = "mlir_tosa"


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
  # A list of artifacts derived from this model.
  artifacts: List[ModelArtifact]

  def __str__(self):
    return self.name


class ArchitectureType(Enum):
  """Type of architecture."""
  CPU = "cpu"
  GPU = "gpu"


@dataclass(frozen=True)
class _ArchitectureInfo(object):
  """Architecture information."""
  type: ArchitectureType
  architecture: str
  microarchitecture: str = ""
  vendor: str = ""

  def __str__(self):
    parts = [
        part for part in (self.vendor, self.architecture,
                          self.microarchitecture) if part != ""
    ]
    return "-".join(parts)


class DeviceArchitecture(_ArchitectureInfo, Enum):
  """Predefined architecture/microarchitecture."""
  # VMVX virtual machine
  VMVX_GENERIC = (ArchitectureType.CPU, "vmvx", "generic")
  # x86_64 CPUs
  X86_64_CASCADELAKE = (ArchitectureType.CPU, "x86_64", "cascadelake")
  # ARM CPUs
  ARM_64_GENERIC = (ArchitectureType.CPU, "arm_64", "generic")
  # RISC-V CPUs
  RV64_GENERIC = (ArchitectureType.CPU, "riscv_64", "generic")
  RV32_GENERIC = (ArchitectureType.CPU, "riscv_32", "generic")
  # Mobile GPUs
  QUALCOMM_ADRENO = (ArchitectureType.GPU, "adreno", "", "qualcomm")
  ARM_VALHALL = (ArchitectureType.GPU, "valhall", "", "arm")
  # CUDA GPUs
  CUDA_SM70 = (ArchitectureType.GPU, "cuda", "sm_70")
  CUDA_SM80 = (ArchitectureType.GPU, "cuda", "sm_80")


@dataclass(frozen=True)
class _HostEnvironmentInfo(object):
  """Environment information of a host.

  The definitions and terms here matches the macros in
  `runtime/src/iree/base/target_platform.h`.

  Note that this is the environment where the runtime "runs". For example:
  ```
  {
    "platform": "linux",
    "architecture": "x86_64"
  }
  ```
  means the runtime will run on a Linux x86_64 host. The runtime might dispatch
  the workloads on GPU or it can be a VM to run workloads compiled in another
  ISA, but those are irrelevant to the information here.
  """
  platform: str
  architecture: str


class HostEnvironment(_HostEnvironmentInfo, Enum):
  """Predefined host environment."""

  LINUX_X86_64 = ("linux", "x86_64")
  ANDROID_ARM_64 = ("android", "arm_64")


@serialization.serializable
@dataclass(frozen=True)
class DeviceSpec(object):
  """Benchmark device specification."""
  id: str
  # Unique name of the device spec.
  name: str
  # Device name. E.g., Pixel-6.
  device_name: str
  # Host environment where the IREE runtime is running. For CPU device type,
  # this is usually the same as the device that workloads are dispatched to.
  # With a separate device, such as a GPU, however, the runtime and dispatched
  # workloads will run on different platforms.
  host_environment: HostEnvironment
  # Architecture of the target device.
  architecture: DeviceArchitecture
  # Device-specific parameters. E.g., 2-big-cores, 4-little-cores.
  # This is for modeling the spec of a heterogeneous processor. Depending on
  # which cores you run, the device has a different spec. Benchmark machines use
  # these parameters to set up the devices. E.g. set CPU mask.
  device_parameters: List[str] = dataclasses.field(default_factory=list)
  # Tags to describe the device spec.
  tags: List[str] = dataclasses.field(default_factory=list)

  def __str__(self):
    return self.name

  @classmethod
  def build(cls,
            id: str,
            device_name: str,
            host_environment: HostEnvironment,
            architecture: DeviceArchitecture,
            device_parameters: Sequence[str] = (),
            tags: Sequence[str] = ()):
    tag_part = f'[{",".join(tags)}]' if len(tags) > 0 else ""
    # Format: <device_name>[<tag>,...]
    name = f"{device_name}{tag_part}"
    return cls(id=id,
               name=name,
               device_name=device_name,
               host_environment=host_environment,
               architecture=architecture,
               device_parameters=list(device_parameters),
               tags=list(tags))


@dataclass
class GenericExecutionBenchmark(object):
  model: Model
  target_device_spec: DeviceSpec

  @classmethod
  def build(cls, model: Model, target_device_spec: DeviceSpec):
    return cls(model=model, target_device_spec=target_device_spec)
