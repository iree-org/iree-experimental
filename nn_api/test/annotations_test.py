# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler import ir
from iree.nn.annotations import *


def run(f):
  print(f"BEGIN: {f.__name__}")
  context = ir.Context()
  with context:
    f()
  print(f"END: {f.__name__}\n")
  return f


def roundtrip(*annots: Annotation):
  attr = Annotation.to_op_attribute(*annots)
  new_annots = Annotation.from_op_attribute(attr)
  new_attr = Annotation.to_op_attribute(*new_annots)
  assert new_attr == attr, f"Roundtrip attributes disagree: {attr} vs {new_attr}"
  print(new_attr)


# CHECK-LABEL: BEGIN: linear_params
@run
def linear_params():
  lpg = LinearParamGroup("foobar", 3)
  lpi = LinearParamItem("foobar", 0)
  # CHECK: [["LinearParams", "foobar", 3 : index], ["LinearParamItem", "foobar", 0 : index]]
  roundtrip(lpg, lpi)
