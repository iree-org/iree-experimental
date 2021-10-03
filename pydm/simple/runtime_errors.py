# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

from simple_lang import SimpleModule

M = SimpleModule(debug=True)


@M.export_pyfunc
def type_error_on_return(condition: bool, true_value: int, false_value: float) -> int:
  if condition:
    return true_value
  else:
    return false_value


@M.export_pyfunc
def unbound_local(condition: bool) -> int:
  if condition:
    r = 1
  return r


class BranchAndCastTest(unittest.TestCase):

  def test_type_error_on_return(self):
    with self.assertRaises(ValueError):
      M.exports.type_error_on_return(0, 1, 2.0)
    self.assertEqual(1, M.exports.type_error_on_return(1, 1, 2.0))

  def test_unbound_local(self):
    self.assertEqual(1, M.exports.unbound_local(True))
    with self.assertRaises(UnboundLocalError):
      M.exports.unbound_local(False)


if __name__ == '__main__':
    unittest.main()
