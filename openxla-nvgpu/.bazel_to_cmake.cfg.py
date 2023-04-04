# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import bazel_to_cmake_targets
import re

DEFAULT_ROOT_DIRS = ["compiler"]

REPO_MAP = {
    # Since this is the @openxla_nvgpu repo, map to empty since all internal
    # targets are of the form "//compiler", not "@openxla_nvgpu//compiler".
    "@openxla_nvgpu": "",
}

class CustomTargetConverter(bazel_to_cmake_targets.TargetConverter):

  def _convert_unmatched_target(self, target: str) -> str:
    """Converts unmatched targets in a repo specific way."""
    # Map //compiler/src/(.*) -> the relative part
    m = re.match(f"^//compiler/src/(.+)", target)
    if m:
      return [self._convert_to_cmake_path(m.group(1))]

    raise ValueError(f"No target matching for {target}")
