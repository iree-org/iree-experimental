# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Builders that help build data types."""

import string
from dataclasses import dataclass
from typing import Dict, List, Union, Sequence

import data_types


@dataclass(frozen=True)
class ModelDataTemplate:
  """ModelData template."""
  id: string.Template
  name: string.Template
  tags: List[Union[str, string.Template]]
  data_format: data_types.DataFormat
  model_id: string.Template
  source_info: str
  tensor_names: List[str]
  tensor_dimensions: List[string.Template]
  source_url: List[string.Template]


@dataclass(frozen=True)
class ModelArtifactTemplate:
  """ModelArtifact template."""
  artifact_type: data_types.ModelArtifactType
  source_url: string.Template


@dataclass(frozen=True)
class ModelTemplate:
  """Model template."""
  id: string.Template
  name: string.Template
  tags: List[Union[str, string.Template]]
  meta_model: data_types.MetaModel
  # Inputs with multiple batch sizes.
  inputs: Dict[int, data_types.ModelData]
  # Ouptuts with multiple batch sizes.
  outputs: Dict[int, data_types.ModelData]
  artifacts: List[ModelArtifactTemplate]


def _substitute_template(value: Union[str, string.Template],
                         **substitutions) -> str:
  if isinstance(value, string.Template):
    return value.substitute(**substitutions)
  return value


def build_batch_model_data(
    template: ModelDataTemplate,
    batch_sizes: Sequence[int]) -> Dict[int, data_types.ModelData]:
  """Build model data with batch sizes by replacing `${batch_size}` in the
  template.

  Args:
    template: model data template with "${batch_size}" to replace.
    batch_sizes: list of batch sizes to generate.

  Returns:
    Map of batch size to model data.
  """

  batch_model_data = {}
  for batch_size in batch_sizes:
    substitute = lambda value: _substitute_template(value=value,
                                                    batch_size=batch_size)
    model_data = data_types.ModelData(
        id=substitute(template.id),
        name=substitute(template.name),
        tags=[substitute(tag) for tag in template.tags],
        data_format=template.data_format,
        model_id=substitute(template.model_id),
        source_info=template.source_info,
        tensor_names=[substitute(name) for name in template.tensor_names],
        tensor_dimensions=[
            substitute(dim) for dim in template.tensor_dimensions
        ],
        source_url=[substitute(url) for url in template.source_url])
    batch_model_data[batch_size] = model_data

  return batch_model_data


def build_batch_models(
    template: ModelTemplate,
    batch_sizes: Sequence[int]) -> Dict[int, data_types.Model]:
  """Build model with batch sizes by replacing `${batch_size}` in the template.

  Args:
    template: model template with "${batch_size}" to replace.
    batch_sizes: list of batch sizes to generate.

  Returns:
    Map of batch size to model.
  """

  batch_models = {}
  for batch_size in batch_sizes:
    substitute = lambda value: _substitute_template(value=value,
                                                    batch_size=batch_size)
    artifacts = []
    for artifact_template in template.artifacts:
      artifacts.append(
          data_types.ModelArtifact(
              artifact_type=artifact_template.artifact_type,
              source_url=substitute(artifact_template.source_url)))
    model = data_types.Model(id=substitute(template.id),
                             name=substitute(template.name),
                             tags=[substitute(tag) for tag in template.tags],
                             meta_model=template.meta_model,
                             input_batch_size=batch_size,
                             inputs=template.inputs[batch_size],
                             outputs=template.outputs[batch_size],
                             artifacts=artifacts)
    batch_models[batch_size] = model

  return batch_models
