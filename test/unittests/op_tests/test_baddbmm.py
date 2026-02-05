# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: E402
    MMTestCase,
    has_zentorch,
    run_tests,
    skip_test_pt_2_0,
    supported_dtypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Baddbmm_Op(MMTestCase):
    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=supported_dtypes
    )
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_baddbmm_variants(self, dtype):

        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d
            ),
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d
            ),
        )

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=['int']
    )
    def test_baddbmm_unsupported_dtype(self, dtype):

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d
            )
        self.assertTrue(
            "zentorch_matmul only supports Float and BFloat16" in str(context.exception)
        )

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=supported_dtypes
    )
    def test_baddbmm_unsupported_dims(self, dtype):

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d.reshape((self.data.b * self.data.m), self.data.n),
                self.data.x3d,
                self.data.y3d,
            )

        self.assertTrue(
            "unsupported dims for self, batch1 and batch2" in str(context.exception)
        )
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.x, self.data.x3d, self.data.x3d
            )
        self.assertTrue(
            "unsupported dims for self, batch1 and batch2" in str(context.exception)
        )

    # TODO: ZENAI-2774 - BFloat16 Precision Loss with Non-Exact Beta Values
    #
    # Issue: When beta is not exactly representable in bfloat16 (e.g., 1.3, 1.4,
    # 0.7), ZenTorch produces different results than PyTorch due to:
    #   1. beta is converted to bf16 before multiplication (loses precision)
    #   2. Matmul accumulation happens in bf16 instead of fp32
    #
    # Original test values (before workaround):
    #   - beta=1.4 (bf16: 1.3984375, error: 0.0015625)
    #   - alpha=1.4, beta=1.3 (bf16: 1.296875, error: 0.003125)
    #
    # Current workaround: Changed beta to 1.0 (exactly representable in bf16)
    #
    # To reproduce manually:
    #   import torch, zentorch
    #   torch.manual_seed(57768)
    #   x = torch.randn(10, 36, 38, dtype=torch.bfloat16)
    #   y = torch.randn(10, 38, 46, dtype=torch.bfloat16)
    #   inp = torch.randn(10, 36, 46, dtype=torch.bfloat16)
    #   ref = torch.baddbmm(inp, x, y, beta=1.3, alpha=1.4)
    #   out = torch.ops.zentorch.zentorch_baddbmm(inp, x, y, beta=1.3, alpha=1.4)
    #   print((ref - out).abs().max())  # Expected: 0.125 (should be < 0.01)

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=supported_dtypes
    )
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_baddbmm_with_kw(self, dtype):
        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, alpha=1.4
            ),
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, alpha=1.4
            ),
        )

        # TODO: Should be removed after the fix for ZENAI-2774
        if dtype == 'bfloat16':
            beta_value = 1.0
        else:
            beta_value = 1.4
        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, beta=beta_value
            ),
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, beta=beta_value
            ),
            atol=1e-2,
            rtol=1e-2,
        )

        # TODO: Should be removed after the fix for ZENAI-2774
        if dtype == 'bfloat16':
            beta_value = 1.0
        else:
            beta_value = 1.3
        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, alpha=1.4, beta=beta_value
            ),
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, alpha=1.4, beta=beta_value
            ),
            atol=1e-2,
            rtol=1e-2,
        )

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=supported_dtypes
    )
    def test_baddbmm_with_zero_alpha(self, dtype):

        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, alpha=0.0
            ),
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, alpha=0.0
            ),
        )

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=['float32']
    )
    def test_float_baddbmm_bfloat16_postop(self):
        bias_as_postop = self.data.input3d.clone().to(torch.bfloat16)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_baddbmm(
                bias_as_postop, self.data.x3d, self.data.y3d
            )
        self.assertTrue(
            "zentorch_matmul only supports Float post ops" in str(context.exception)
        )

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=['bfloat16']
    )
    def test_bfloat16_baddbmm_int_postop(self):
        self.skip_if_bfloat16_unsupported_hardware()
        bias_as_postop = self.data.input3d.clone().to(torch.int)
        with self.assertRaises(RuntimeError) as context_int:
            torch.ops.zentorch.zentorch_baddbmm(
                bias_as_postop, self.data.x3d, self.data.y3d
            )
        self.assertTrue(
            "zentorch_matmul only supports BFloat16 post ops"
            in str(context_int.exception)
        )

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=['int']
    )
    def test_int_baddbmm_postop(self):
        bias_as_postop = self.data.x3d.clone().to(torch.int)
        with self.assertRaises(RuntimeError) as context_int:
            torch.ops.zentorch.zentorch_baddbmm(
                bias_as_postop, self.data.x3d, self.data.x3d
            )
        self.assertTrue(
            "zentorch_matmul only supports Float and BFloat16"
            in str(context_int.exception)
        )


if __name__ == "__main__":
    run_tests()
