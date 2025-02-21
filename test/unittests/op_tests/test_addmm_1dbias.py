# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from parameterized import parameterized
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    supported_dtypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Addmm_1dbias(Zentorch_TestCase):

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_1dbias_incorrect_dims(self, dtype):

        self.data.create_unittest_data(dtype)

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm_1dbias(
                self.data.x, self.data.x, self.data.x
            )
        self.assertTrue(
            "unsupported dims for self, mat1 and mat2" in str(context.exception)
        )


if __name__ == "__main__":
    run_tests()
