# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, List, Sequence

import io
import jax
from jax.lib import xla_bridge
from jax.lib import xla_client
import numpy as np

from iree.compiler.api import xla as iree_xla

from . import ir_builder
from .iree_imports import *


class IreeDevice:

  def __init__(self, client):
    self.id = 0
    self.host_id = 0
    self.process_index = 0
    self.platform = "toy"
    self.device_kind = "toy device"
    self.client = client

  def __str__(self) -> str:
    return "IreeDevice"

  def transfer_to_infeed(self, literal: Any):
    raise NotImplemented("transfer_to_infeed")

  def transfer_from_outfeed(self, shape: xla_client.Shape):
    raise NotImplemented("transfer_to_outfeed")

  def live_buffers(self) -> List['IreeBuffer']:
    raise NotImplemented("live_buffers")


class IreeBuffer(xla_client.DeviceArrayBase):

  def __init__(self, client, device, npy_value):
    self.client = client
    self._device = device
    self._npy_value = np.asarray(npy_value)

  def to_py(self) -> np.ndarray:
    return self._npy_value

  def to_iree(self):
    return self._npy_value

  # _device: Optional[Device]
  # aval: Any
  # weak_type: Optional[bool]
  # _lazy_expr: Any
  # @property
  # def device_buffer(self: _T) -> _T: ...
  # shape: Tuple[int, ...]
  # dtype: np.dtype
  # size: int
  # ndim: int
  # _value: np.ndarray
  # def copy_to_device(self, dst_device: Device) -> DeviceArray: ...
  # def on_device_size_in_bytes(self) -> int: ...
  # def delete(self) -> None: ...
  # def block_until_ready(self) -> DeviceArray: ...
  # def copy_to_host_async(self) -> _Status: ...
  # def xla_shape(self) -> Shape: ...
  # def xla_dynamic_shape(self) -> Shape: ...
  # client: Client
  # def device(self) -> Device: ...
  # def platform(self) -> str: ...
  # def is_deleted(self) -> bool: ...
  # def unsafe_buffer_pointer(self) -> Any: ...
  # __cuda_array_interface__: Dict[str, Any]
  # traceback: Traceback
  # def clone(self) -> DeviceArray: ...


class IreeExecutable:

  def __init__(self, client, devices, module_object, function_name):
    self.client = client
    self.traceback = None
    self.fingerprint = None
    self._devices = devices
    self.module_object = module_object
    self.function_name = function_name

  def local_devices(self) -> List[IreeDevice]:
    return self._devices

  def execute(self, arguments: Sequence[IreeBuffer]) -> List[IreeBuffer]:
    inputs = [arg.to_iree() for arg in arguments]
    outputs = self.module_object[self.function_name](*inputs)
    # TODO: Have a way to just have it always return the list, regardless of
    # arity.
    if not isinstance(outputs, list):
      outputs = [outputs]
    return [
        IreeBuffer(self.client, self._devices[0], output) for output in outputs
    ]

  # def local_logical_device_ids(self) -> List[Tuple[int, int]]: ...
  # def size_of_generated_code_in_bytes(self) -> int: ...
  # def delete(self) -> None: ...
  # def execute_sharded_on_local_devices(
  #     self,
  #     arguments: Sequence[List[DeviceArray]]) -> List[List[DeviceArray]]: ...
  # def hlo_modules(self) -> List[HloModule]: ...
  # def keep_alive(self) -> None: ...


class IreeClient:

  def __init__(self,
               *,
               compile_target_backends: Sequence[str] = ("cpu",),
               runtime_driver: str = "dylib"):
    self.platform = "iree_simple"
    self.platform_version = "0.0.1"
    self.runtime_type = "iree"
    self.iree_config = iree_runtime.system_api.Config(runtime_driver)
    self.compiler_options = compiler_driver.CompilerOptions()
    self.compiler_options.set_input_dialect_mhlo()
    for target_backend in compile_target_backends:
      self.compiler_options.add_target_backend(target_backend)
    self._devices = [IreeDevice(self)]

  def process_index(self) -> int:
    return 0

  def device_count(self) -> int:
    return len(self._devices)

  def devices(self) -> List[IreeDevice]:
    return self._devices

  def local_devices(self) -> List[IreeDevice]:
    return self._devices

  def local_device_count(self) -> int:
    return len(self._devices)

  def get_default_device_assignment(
      self,
      num_replicas: int,
      num_partitions: int = 1) -> List[List[IreeDevice]]:
    if num_replicas != 1 or num_partitions != 1:
      raise NotImplemented("Only single-device computations implemented")
    return [[self._devices[0]]]

  def compile(self, computation: xla_client.XlaComputation,
              compile_options: xla_client.CompileOptions) -> IreeExecutable:
    # Loop it through the XLA->MLIR converter.
    # TODO: Do a direct conversion instead of via XLA and MLIR ASM.
    hlo_text = computation.as_hlo_text()
    iree_asm = iree_xla.compile_str(
        hlo_text, import_only=True, import_format="HLO_TEXT")
    b = ir_builder.Builder(parse_asm=iree_asm)

    # Compile it.
    with b.context:
      pm = passmanager.PassManager()
      compiler_driver.build_iree_vm_pass_pipeline(self.compiler_options, pm)
      pm.run(b.input_module)
      bytecode_io = io.BytesIO()
      compiler_driver.translate_module_to_vm_bytecode(self.compiler_options,
                                                      b.input_module,
                                                      bytecode_io)
      iree_binary = bytecode_io.getbuffer()

    # Load it into the runtime.
    # TODO: Have a public API for this (not in 'binding').
    vm_module = iree_runtime.binding.VmModule.from_flatbuffer(iree_binary)
    module_object = iree_runtime.load_vm_module(vm_module, self.iree_config)
    return IreeExecutable(self, self._devices, module_object, "main")

  def buffer_from_pyval(
      self,
      argument: Any,
      device: IreeDevice,
      force_copy: bool = True,
      host_buffer_semantics: xla_client.HostBufferSemantics = xla_client
      .HostBufferSemantics.ZERO_COPY
  ) -> IreeBuffer:
    # TODO: IREE's python API will accept a numpy array directly but may
    # want to explicitly construct a lower level BufferView to avoid copies.
    return IreeBuffer(self, device, np.array(argument, copy=True))

  # def live_buffers(self) -> List[Buffer]: ...
  # def live_executables(self) -> List[Executable]: ...
  # def host_id(self) -> int: ...

  # def create_channel_handle(self) -> ChannelHandle: ...
  # def create_device_to_host_channel_handle(self) -> ChannelHandle: ...
  # def create_host_to_device_channel_handle(self) -> ChannelHandle: ...

  # def serialize_executable(self, executable: Executable) -> bytes: ...
  # def deserialize_executable(
  #     self, serialized: bytes,
  #     options: CompileOptions) -> Executable: ...
  # # TODO(skyewm): remove when jax stop providing hlo_module
  # def deserialize_executable(
  #     self, serialized: bytes,
  #     hlo_module: HloModule,
  #     options: CompileOptions) -> Executable: ...
  # def heap_profile(self) -> bytes: ...
  # def defragment(self) -> _Status: ...
  # def emit_python_callback(
  #     self, callable: Callable, builder: XlaBuilder, operands: Sequence[XlaOp],
  #     results_shapes: Sequence[Shape],
  #     operand_layouts: Optional[Sequence[Shape]] = ...,
  #     has_side_effects: bool = ...) -> Tuple[XlaOp, Any]: ...


def iree_client_factory():
  return IreeClient()


def register_backend(priority: int = 1000):
  xla_bridge.register_backend_factory(
      "iree", iree_client_factory, priority=priority)
