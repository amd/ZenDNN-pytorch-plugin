# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    RmsNormTestCase,
    supported_dtypes,
    has_zentorch,
    run_tests,
    zentorch,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_RMS_Norm(RmsNormTestCase):
    @RmsNormTestCase.hypothesis_params_rms_norm_itr(
        dtype_list=supported_dtypes,
    )
    def test_rms_norm(self, dtype):
        epsilon = 1e-6
        dtype_torch = getattr(torch, dtype)

        input = self.data.rms_input
        weight = self.data.rms_weight
        hidden_size = self.data.hidden_size

        # reference RMSNorm
        rms_norm = nn.RMSNorm(hidden_size, eps=epsilon).to(dtype_torch)
        rms_norm.weight = nn.Parameter(weight)
        ref_output = rms_norm(input)

        # zen RMSNorm returns normalized output in a new tensor
        zen_output = torch.ops.zentorch.zentorch_rms_norm(
            input, weight, epsilon
        )
        self.assertEqual(
            ref_output,
            zen_output,
            atol=1e-3,
            rtol=1e-2,
            msg=f"RMSNorm output mismatch for batch_size={self.data.batch_size}, "
            f"hidden_size={hidden_size}",
        )

    @RmsNormTestCase.hypothesis_params_rms_norm_itr(
        dtype_list=supported_dtypes
    )
    def test_add_rms_norm(self, dtype):
        epsilon = 1e-6
        dtype_torch = getattr(torch, dtype)

        # Clone tensors because zentorch_add_rms_norm_ operates in-place
        input = self.data.rms_input.clone()
        weight = self.data.rms_weight
        residual = self.data.rms_residual.clone()
        hidden_size = self.data.hidden_size

        rms_norm = nn.RMSNorm(hidden_size, eps=epsilon).to(dtype_torch)
        rms_norm.weight = nn.Parameter(weight)
        ref_residual = input + residual
        ref_output = rms_norm(ref_residual)

        torch.ops.zentorch.zentorch_add_rms_norm_(
            input, weight, residual, epsilon
        )
        zen_output, zen_residual = input, residual
        self.assertEqual(
            ref_output,
            zen_output,
            atol=1e-3,
            rtol=1e-2,
            msg=f"Output mismatch for batch_size={self.data.batch_size}, "
            f"hidden_size={hidden_size}",
        )
        self.assertEqual(
            ref_residual,
            zen_residual,
            atol=1e-3,
            rtol=1e-2,
            msg=f"Residual mismatch for batch_size={self.data.batch_size}, "
            f"hidden_size={hidden_size}",
        )


if __name__ == "__main__":
    run_tests()
