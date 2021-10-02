# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from simple_lang import SimpleModule

M = SimpleModule()


@M.export_pyfunc
def return_arg(a: int, b: int, c: int) -> int:
  #return a if b else c
  if b:
    return a
  else:
    return c


M.save("/tmp/basic_sample.vmfb")
print(M.loaded_module.return_arg(5, 0, 1))
