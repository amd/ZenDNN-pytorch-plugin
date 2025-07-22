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
    MMTestCase,
    has_zentorch,
    run_tests,
    supported_dtypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_MM_Silu(MMTestCase):
    @parameterized.expand(supported_dtypes)
    # Switching to Hypothesis exposes more issues, so the existing methods are retained.
    # Please refer ZENAI-1968 for details
    # @MMTestCase.hypothesis_params_mm_itr(
    #     dtype_list=supported_dtypes
    # )
    @torch.inference_mode()
    def test_mm_silu(self, dtype):
        self.data.create_unittest_data(dtype)
        native_output = torch.nn.functional.silu(torch.matmul(self.data.x, self.data.y))
        zentorch_output = torch.ops.zentorch.zentorch_mm_silu(self.data.x, self.data.y)

        self.assertEqual(native_output, zentorch_output)


if __name__ == "__main__":
    run_tests()
