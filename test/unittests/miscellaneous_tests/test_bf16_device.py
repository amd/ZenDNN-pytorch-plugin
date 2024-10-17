# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import TestCase, has_zentorch, run_tests, zentorch  # noqa: 402


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_BF16_Device(TestCase):
    @unittest.skipIf(
        not zentorch._C.is_bf16_supported(), "CPU does not support AVX512 BF16."
    )
    def test_bf16_device(self):
        self.assertTrue(zentorch._C.is_bf16_supported(), "CPU supports AVX512 BF16.")


if __name__ == "__main__":
    run_tests()
