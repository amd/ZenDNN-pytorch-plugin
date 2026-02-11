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
    AddmmTestCase,
    has_zentorch,
    run_tests,
    supported_dtypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Addmm_SiLU_Mul(AddmmTestCase):

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_addmm_silu_mul(self, dtype):
        bias = self.data.input.clone()
        native_output = (
            torch.nn.functional.silu(torch.addmm(bias, self.data.x, self.data.y))
            * self.data.input
        )
        zentorch_output = torch.ops.zentorch.zentorch_addmm_silu_mul(
            bias, self.data.x, self.data.y, self.data.input
        )

        # TODO
        # Tensor Generation and Tolerance Calculation will be aligned with ZenDNN library in future.

        self.assertEqual(native_output, zentorch_output, atol=1e-2, rtol=1e-2)

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_addmm_silu_mul_mismatched_dimensions(self, dtype):
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm_silu_mul(
                self.data.input,
                self.data.x,
                self.data.y,
                torch.reshape(
                    self.data.input,
                    (1, list(self.data.input.shape)[0], list(self.data.input.shape)[1]),
                ),
            )
        self.assertTrue(
            "unsupported dims for mat1, mat2 and post op buffer"
            in str(context.exception)
        )

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_addmm_silu_mul_mismatched_sizes(self, dtype):
        # The test will not fail when k == n
        # When K == N, Dimensions will be compatible even after reshaping
        if self.data.k != self.data.n:
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zentorch_addmm_silu_mul(
                    self.data.input, self.data.x, self.data.y, self.data.x
                )
            self.assertTrue(
                "unsupported shapes for mat1, mat2 and post op buffer"
                in str(context.exception)
            )


if __name__ == "__main__":
    run_tests()
