# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
from importlib import metadata
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import(  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    zentorch,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_ZenTorch_Version(Zentorch_TestCase):
    def test_zentorch_version(self):
        self.assertTrue(zentorch.__version__, metadata.version("zentorch"))


if __name__ == "__main__":
    run_tests()
