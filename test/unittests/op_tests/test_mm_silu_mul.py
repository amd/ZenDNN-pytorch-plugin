# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
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
class Test_MM_SiLU_Mul(MMTestCase):
    @parameterized.expand(supported_dtypes)
    # Switching to Hypothesis exposes more issues, so the existing methods are retained.
    # Please refer ZENAI-1969 for details
    # @MMTestCase.hypothesis_params_mm_itr(
    #     dtype_list=supported_dtypes
    # )
    @torch.inference_mode()
    def test_mm_silu_mul(self, dtype):
        self.data.create_unittest_data(dtype)
        native_output = (
            torch.nn.functional.silu(torch.matmul(self.data.x, self.data.y))
            * self.data.input
        )
        zentorch_output = torch.ops.zentorch.zentorch_mm_silu_mul(
            self.data.x, self.data.y, self.data.input
        )
        self.assertEqual(native_output, zentorch_output)

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_mm_silu_mul_mismatched_dimensions(self, dtype):
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_mm_silu_mul(
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

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=supported_dtypes,
    )
    @torch.inference_mode()
    def test_mm_silu_mul_mismatched_sizes(self, dtype):
        # The test will not fail when k == n
        # When K == N, Dimensions will be compatible even after reshaping
        if self.data.k != self.data.n:
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zentorch_mm_silu_mul(
                    self.data.x, self.data.y, self.data.x
                )
            self.assertTrue(
                "unsupported shapes for mat1, mat2 and post op buffers"
                in str(context.exception)
            )


if __name__ == "__main__":
    run_tests()
