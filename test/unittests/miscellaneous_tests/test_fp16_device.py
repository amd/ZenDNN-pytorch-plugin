# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    zentorch,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Fp16_Device(Zentorch_TestCase):
    @unittest.skipIf(
        not zentorch._C.is_fp16_supported(), "CPU does not support AVX512 FP16."
    )
    def test_fp16_device(self):
        self.assertTrue(zentorch._C.is_fp16_supported(), "CPU supports AVX512 FP16.")


if __name__ == "__main__":
    run_tests()
