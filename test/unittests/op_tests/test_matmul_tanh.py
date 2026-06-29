# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    MMTestCase,
    AddmmTestCase,
    has_zentorch,
    run_tests,
    reset_dynamo,
    supported_dtypes,
    update_supported_dtypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_MM_Tanh(MMTestCase):
    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=update_supported_dtypes(supported_dtypes, "zentorch_mm")
    )
    @torch.inference_mode()
    def test_mm_tanh(self, dtype):
        tol = 1e-2 if dtype == "float16" else 1e-5
        native_output = torch.tanh(torch.mm(self.data.x, self.data.y))
        zentorch_output = torch.ops.zentorch.zentorch_mm_tanh(self.data.x, self.data.y)
        self.assertEqual(native_output, zentorch_output, atol=tol, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
