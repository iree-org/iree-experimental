import os
import sys

import lit.formats
import lit.util

import lit.llvm

# Configuration file for the 'lit' test runner.
lit.llvm.initialize(lit_config, config)

# name: The name of this test suite.
config.name = 'TFLITEHUB'

config.test_format = lit.formats.ShTest()

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.py']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

#config.use_default_substitutions()
config.excludes = [
  'lit.cfg.py',
  'lit.site.cfg.py',
  'test_util.py',
]

config.substitutions.extend([
    ('%PYTHON', sys.executable),
])

project_root = os.path.dirname(os.path.dirname(__file__))
config.environment["FILECHECK_OPTS"] = "--dump-input=fail"
