# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from simple_lang import jit
from simple_lang import SimpleModule

# Due to restrictions in the way that simple_lang is implemented, recursive
# modules need to be defined at the top level like this.
FibRecursive = SimpleModule(debug=2)


@FibRecursive.export_pyfunc
def fib_recursive(n: int) -> int:
  if n <= 1:
    return n
  return fib_recursive(n - 1) + fib_recursive(n - 2)


class FibonacciTest(unittest.TestCase):

  def test_recursive(self):
    print(FibRecursive.exports.fib_recursive(5))

  def test_list(self):
    @jit(debug=0)
    def compute(n: int) -> int:
      values = [0] * (n + 2)
      values[0] = 0
      values[1] = 1
      i = 2
      while i <= n:  # TODO: Upgrade to for...range
        values[i] = values[i - 1] + values[i - 2]
        i = i + 1  # TODO: Support AugAssign
      return values[n]
    print("FIB_LIST:", compute(20))

  def test_spaceopt(self):
    @jit(debug=0)
    def compute(n: int) -> int:
      a = 0
      b = 1
      if n == 0:
        return a
      i = 2
      while i <= n:  # TODO: Upgrade to for...range
        c = a + b
        a = b
        b = c
        i = i + 1  # TODO: Support AugAssign
      return b
    print("FIB_OPT:", compute(20))


if __name__ == '__main__':
    unittest.main()
