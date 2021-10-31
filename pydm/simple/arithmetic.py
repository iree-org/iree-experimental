# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from simple_lang import jit


class ArithmeticTest(unittest.TestCase):

  def test_add_int_args(self):
    @jit
    def add(a: int, b: int) -> int:
      return a + b
    self.assertEqual(7, add(3, 4))

  def test_sub_int_args(self):
    @jit
    def add(a: int, b: int) -> int:
      return a - b
    self.assertEqual(-1, add(3, 4))

  def test_int_arithmetic(self):
    @jit(debug=2)
    def compute(a: int, b: int, c: int) -> int:
      if a - b:
        return 3 * a + 22 * b
      else:
        return 4 * c
    self.assertEqual(103, compute(5, 4, 10))
    self.assertEqual(40, compute(4, 4, 10))

  # TODO: Something wrong with float arithmetic
  def test_float_arithmetic(self):
    @jit
    def add(a: float, b: float) -> float:
      return 3.0 * a - 2.0 * b
    self.assertEqual(9, add(3.0, 5.0))


if __name__ == '__main__':
    unittest.main()
