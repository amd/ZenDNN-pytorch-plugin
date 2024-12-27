# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
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
class Test_AVX512_Device(Zentorch_TestCase):
    @unittest.skipIf(
        not zentorch._C.is_avx512_supported(),
        "CPU does not support AVX512 instructions.",
    )
    def test_avx512_device(self):
        self.assertTrue(
            zentorch._C.is_avx512_supported(), "CPU supports AVX512 instructions."
        )


if __name__ == "__main__":
    run_tests()
