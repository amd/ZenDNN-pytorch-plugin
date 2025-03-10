# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
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
    skip_test_pt_2_0,
    supported_dtypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_MM_Op(MMTestCase):
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=supported_dtypes
    )
    def test_mm_variants(self, dtype):
        # mm
        self.assertEqual(
            torch._C._VariableFunctions.mm(self.data.x, self.data.y),
            torch.ops.zentorch.zentorch_mm(self.data.x, self.data.y),
        )
        self.assertEqual(
            torch.matmul(self.data.x, self.data.y),
            torch.ops.zentorch.zentorch_mm(self.data.x, self.data.y),
        )
        self.assertEqual(
            torch.mm(self.data.x, self.data.y),
            torch.ops.zentorch.zentorch_mm(self.data.x, self.data.y),
        )

        self.assertEqual(
            self.data.x @ self.data.y,
            torch.ops.zentorch.zentorch_mm(self.data.x, self.data.y),
        )

        self.assertEqual(
            torch.mul(self.data.A, self.data.B),
            torch.ops.zentorch.zentorch_mm(self.data.A, self.data.B),
        )

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=supported_dtypes
    )
    def test_mm_mismatched_dimensions(self, dtype):
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_mm(
                self.data.x,
                torch.reshape(
                    self.data.x,
                    (1, list(self.data.x.shape)[0], list(self.data.x.shape)[1]),
                ),
            )
        self.assertTrue("unsupported dims for self and mat2" in str(context.exception))
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_mm(self.data.x3d, self.data.x3d)
        self.assertTrue("unsupported dims for self and mat2" in str(context.exception))

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=["int"])
    def test_mm_unsupported_dtype(self, dtype):

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_mm(self.data.x, self.data.y)
        self.assertTrue(
            "zentorch_matmul only supports Float and BFloat16" in str(context.exception)
        )

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=supported_dtypes
    )
    def test_mm_relu(self, dtype):

        # mm->relu
        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.mm(self.data.x, self.data.y)
            ),
            torch.ops.zentorch.zentorch_mm_relu(self.data.x, self.data.y),
        )


if __name__ == "__main__":
    run_tests()
