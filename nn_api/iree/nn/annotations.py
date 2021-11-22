# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Annotations are persisted on operations in the program via attributes.

The Python interpretation of them is established here.
"""

from typing import Container, List, Optional, Sequence

from iree.compiler.ir import (
    Attribute,
    ArrayAttr,
    IndexType,
    IntegerAttr,
    StringAttr,
)


class AnnotationMeta(type):
  """Metaclass for annotations.

  Tracks each created subclass in the registry.
  """
  REGISTRY = dict()

  def __new__(mcs, name, bases, attrs, **kwargs):
    new_class = super().__new__(mcs, name, bases, attrs)

    # Add it to the registry (unless if the base class).
    if name != "Annotation":
      try:
        ident = new_class.IDENT
      except AttributeError:
        raise TypeError("Annotation class must have an IDENT attribute")
      try:
        accepts_idents = new_class.ACCEPTS_IDENTS
      except AttributeError:
        accepts_idents = (ident,)

      for new_ident in accepts_idents:
        if new_ident in mcs.REGISTRY:
          raise TypeError(
              f"New Annotation class defines ident {new_ident} which "
              f"is already bound to {mcs.REGISTRY[new_ident]}")
        mcs.REGISTRY[new_ident] = new_class

    return new_class


class Annotation(metaclass=AnnotationMeta):
  """Base class for all annotations."""

  @staticmethod
  def to_op_attribute(*annots: "Annotation") -> Attribute:
    """Serializes a sequence of annotations.

    This is used to construct the final `iree.nn.annot` operation attribute.
    """
    return ArrayAttr.get([annot.to_attribute() for annot in annots])

  @staticmethod
  def from_op_attribute(annot_attr: Attribute) -> Sequence["Annotation"]:
    """Deserializes a sequence of annotations."""
    array_attr = ArrayAttr(annot_attr)
    results = []
    for entry_attr in array_attr:
      results.append(Annotation.from_attribute(entry_attr))
    return results

  def to_attribute(self) -> Attribute:
    """Gets an Attribute that represents the overall state.

    Every attribute is a list, starting with the IDENT.
    """
    return ArrayAttr.get([StringAttr.get(self.IDENT)] + self._get_attr_state())

  @staticmethod
  def from_attribute(attr: Attribute) -> "Annotation":
    entries = list(ArrayAttr(attr))
    ident = StringAttr(entries[0])
    try:
      annot_class = AnnotationMeta.REGISTRY[ident.value]
    except KeyError:
      raise ValueError(f"No registered annotation class for {ident}")
    return annot_class._from_attr_state(entries[1:])

  def _get_attr_state(self) -> List[Attribute]:
    raise NotImplementedError

  @classmethod
  def _from_attr_state(cls, state: List[Attribute]) -> "Annotation":
    raise NotImplementedError


class LinearParamGroup(Annotation):
  """Represents a linearized set of parameters in a model.

  This attribute is applied to the first parameter global which forms a linear
  sequence of named parameters. Each global must also have a LinearParamItem
  annotation.
  """
  IDENT = "LinearParams"

  def __init__(self, name: str, arity: int):
    self.name = name
    self.arity = arity

  def _get_attr_state(self) -> List[Attribute]:
    return [
        StringAttr.get(self.name),
        IntegerAttr.get(IndexType.get(), self.arity)
    ]

  @classmethod
  def _from_attr_state(cls, state: List[Attribute]) -> "Annotation":
    return cls(
        name=StringAttr(state[0]).value,
        arity=IntegerAttr(state[1]).value)


class LinearParamItem(Annotation):
  IDENT = "LinearParamItem"

  def __init__(self, name: str, index: int):
    self.name = name
    self.index = index

  def _get_attr_state(self) -> List[Attribute]:
    return [
        StringAttr.get(self.name),
        IntegerAttr.get(IndexType.get(), self.index)
    ]

  @classmethod
  def _from_attr_state(cls, state: List[Attribute]) -> "Annotation":
    return cls(
        name=StringAttr(state[0]).value,
        index=IntegerAttr(state[1]).value)
