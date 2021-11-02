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
    @jit(debug=0)
    def compute(count: int, index: int) -> int:
      lst = [1, 2, 3] * count
      return lst[index]
    full_list = [compute(5, i) for i in range(15)]
    self.assertEqual(full_list, [1, 2, 3] * 5)
    with self.assertRaises(IndexError):
      compute(5, 15)
    with self.assertRaises(IndexError):
      compute(0, 0)
    with self.assertRaises(IndexError):
      compute(-1, 0)

  def test_list_setitem(self):
    @jit(debug=2)
    def compute(write: int, value: int, read: int) -> int:
      lst = [1, 2, 3]
      lst[write] = value
      return lst[read]
    self.assertEqual(99, compute(1, 99, -2))
    self.assertEqual(100, compute(-2, 100, 1))
    with self.assertRaises(IndexError):
      compute(3, 100, 0)
    with self.assertRaises(IndexError):
      compute(-4, 100, 0)


if __name__ == '__main__':
    unittest.main()
