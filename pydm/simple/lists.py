# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from simple_lang import jit


class ListsTest(unittest.TestCase):

  def test_construct_literal_index(self):
    @jit(debug=0)
    def compute(a: int, b: int) -> int:
      lst = [a, b, 9]
      return lst[1]
    self.assertEqual(7, compute(1, 7))

  def test_construct_literal_index_negative(self):
    @jit(debug=0)
    def compute(a: int, b: int, index: int) -> int:
      lst = [a, b, 9]
      return lst[index]
    self.assertEqual(7, compute(1, 7, -2))

  def test_construct_literal_index_error(self):
    @jit(debug=0)
    def compute(a: int, b: int, index: int) -> int:
      lst = [a, b, 9]
      return lst[index]
    with self.assertRaises(IndexError):
      compute(1, 7, -4)
    with self.assertRaises(IndexError):
      compute(1, 7, 4)

  def test_construct_literal_index_imm(self):
    @jit(debug=0)
    def compute(a: int, b: int) -> int:
      lst = [a, b, 9]
      return lst[-1]
    self.assertEqual(9, compute(1, 7))

  def test_list_multiply(self):
    @jit(debug=2)
    def compute(count: int, index: int) -> int:
      lst = [1, 2, 3] * count
      return lst[index]
    compute(5, 4)

if __name__ == '__main__':
    unittest.main()
