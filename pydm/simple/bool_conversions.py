# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from simple_lang import SimpleModule

M = SimpleModule(debug=True)


@M.export_pyfunc
def object_as_bool_int(condition: int, true_value: int, false_value: int) -> int:
  if condition:
    return true_value
  else:
    return false_value


@M.export_pyfunc
def object_as_bool_bool(condition: bool, true_value: int, false_value: int) -> int:
  if condition:
    return true_value
  else:
    return false_value


@M.export_pyfunc
def object_as_bool_float(condition: float, true_value: int, false_value: int) -> int:
  if condition:
    return true_value
  else:
    return false_value


@M.export_pyfunc
def type_error_on_return(condition: bool, true_value: int, false_value: float) -> int:
  if condition:
    return true_value
  else:
    return false_value


class BranchAndCastTest(unittest.TestCase):

  def test_object_as_bool_int(self):
    self.assertEqual(2, M.exports.object_as_bool_int(0, 1, 2))
    self.assertEqual(1, M.exports.object_as_bool_int(1, 1, 2))

  def test_object_as_bool_bool(self):
    self.assertEqual(2, M.exports.object_as_bool_int(False, 1, 2))
    self.assertEqual(1, M.exports.object_as_bool_int(True, 1, 2))

  def test_object_as_bool_float(self):
    self.assertEqual(2, M.exports.object_as_bool_float(0.0, 1, 2))
    self.assertEqual(1, M.exports.object_as_bool_float(1.0, 1, 2))

  def test_type_error_on_return(self):
    with self.assertRaises(ValueError):
      M.exports.type_error_on_return(0, 1, 2.0)
    self.assertEqual(1, M.exports.type_error_on_return(1, 1, 2.0))


if __name__ == '__main__':
    unittest.main()
