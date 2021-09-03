# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Concentrates IREE related imports in one place.

Different environments locate these in different ways, and indirecting through
this module provides one place to replace them.
"""

from iree.compiler import ir
from iree.compiler import passmanager
from iree.compiler.api import driver as compiler_driver
from iree.compiler.dialects import builtin, chlo, mhlo, std
from iree.compiler.dialects import iree as iree_dialect
from iree import runtime as iree_runtime

__all__ = [
    "compiler_driver",
    "ir",
    "iree_runtime",
    "passmanager",
    # Dialects.
    "builtin",
    "chlo",
    "iree_dialect",
    "mhlo",
    "std",
]
