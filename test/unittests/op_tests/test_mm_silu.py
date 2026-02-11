# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
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
    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_mm_silu(self, dtype):
        native_output = torch.nn.functional.silu(torch.matmul(self.data.x, self.data.y))
        zentorch_output = torch.ops.zentorch.zentorch_mm_silu(self.data.x, self.data.y)

        # TODO
        # Tensor Generation and Tolerance Calculation will be aligned with ZenDNN library in future.

        self.assertEqual(native_output, zentorch_output, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
