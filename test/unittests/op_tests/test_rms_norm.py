# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
from parameterized import parameterized
from itertools import product
import torch
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    zentorch,
    freeze_opt,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_RMS_Norm(Zentorch_TestCase):
    def setUp(self):
        super().setUp()
        self.epsilon = 1e-6
        self.batch_sizes = [2]
        self.hidden_sizes = [64]

    @parameterized.expand(freeze_opt)
    def test_rms_norm(self, freeze_opt):
        for batch_size, hidden_size in product(self.batch_sizes, self.hidden_sizes):
            input = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16)
            weight = torch.randn(hidden_size, dtype=torch.bfloat16)

            # reference RMSNorm
            rms_norm = nn.RMSNorm(hidden_size, eps=self.epsilon).bfloat16()
            rms_norm.weight = nn.Parameter(weight)
            ref_output = rms_norm(input)

            # zen RMSNorm returns normalized output in a new tensor
            zen_output = torch.ops.zentorch.zentorch_rms_norm(
                input, weight, self.epsilon
            )
            self.assertEqual(
                ref_output,
                zen_output,
                atol=1e-3,
                rtol=1e-2,
                msg=f"RMSNorm output mismatch for batch_size={batch_size}, "
                f"hidden_size={hidden_size}",
            )

    @parameterized.expand(freeze_opt)
    def test_add_rms_norm(self, freeze_opt):
        for batch_size, hidden_size in product(self.batch_sizes, self.hidden_sizes):
            input = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16)
            weight = torch.randn(hidden_size, dtype=torch.bfloat16)
            residual = torch.randn(batch_size, hidden_size, dtype=torch.bfloat16)

            rms_norm = nn.RMSNorm(hidden_size, eps=self.epsilon).bfloat16()
            rms_norm.weight = nn.Parameter(weight)
            ref_residual = input + residual
            ref_output = rms_norm(ref_residual)

            torch.ops.zentorch.zentorch_add_rms_norm_(
                input, weight, residual, self.epsilon
            )
            zen_output, zen_residual = input, residual
            self.assertEqual(
                ref_output,
                zen_output,
                atol=1e-3,
                rtol=1e-2,
                msg=f"Output mismatch for batch_size={batch_size}, "
                f"hidden_size={hidden_size}",
            )
            self.assertEqual(
                ref_residual,
                zen_residual,
                atol=1e-3,
                rtol=1e-2,
                msg=f"Residual mismatch for batch_size={batch_size}, "
                f"hidden_size={hidden_size}",
            )


if __name__ == "__main__":
    run_tests()
