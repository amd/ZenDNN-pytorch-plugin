# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from parameterized import parameterized
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: E402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    skip_test_pt_2_0,
    supported_dtypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Baddbmm_Op(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_baddbmm_variants(self, dtype):

        self.data.create_data(dtype)
        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d
            ),
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d
            ),
        )

    @parameterized.expand([("int",)])
    def test_baddbmm_unsupported_dtype(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d
            )

        self.assertTrue(
            "zentorch_matmul only supports Float and BFloat16" in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    def test_baddbmm_unsupported_dims(self, dtype):

        self.data.create_data(dtype)
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

    @parameterized.expand(supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_baddbmm_with_kw(self, dtype):
        self.data.create_data(dtype)
        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, alpha=1.4
            ),
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, alpha=1.4
            ),
        )

        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, beta=1.4
            ),
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, beta=1.4
            ),
            atol=1e-2,
            rtol=1e-2,
        )

        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, alpha=1.4, beta=1.3
            ),
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, alpha=1.4, beta=1.3
            ),
            atol=1e-2,
            rtol=1e-2,
        )

    @parameterized.expand(supported_dtypes)
    def test_baddbmm_with_zero_alpha(self, dtype):

        self.data.create_data(dtype)
        self.assertEqual(
            torch._C._VariableFunctions.baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, alpha=0.0
            ),
            torch.ops.zentorch.zentorch_baddbmm(
                self.data.input3d, self.data.x3d, self.data.y3d, alpha=0.0
            ),
        )

    def test_float_baddbmm_bfloat16_postop(self):
        self.data.create_data("float32")
        with self.assertRaises(RuntimeError) as context:
            bias_as_postop = self.data.input3d.clone().to(torch.bfloat16)
            torch.ops.zentorch.zentorch_baddbmm(
                bias_as_postop, self.data.x3d, self.data.y3d
            )

        self.assertTrue(
            "zentorch_matmul only supports Float and BFloat16" in str(context.exception)
        )

    def test_bfloat16_baddbmm_int_postop(self):
        self.skip_if_bfloat16_unsupported_hardware()
        self.data.create_data("bfloat16")
        with self.assertRaises(RuntimeError) as context_int:
            bias_as_postop = self.data.input3d.clone().to(torch.int)
            torch.ops.zentorch.zentorch_baddbmm(
                bias_as_postop, self.data.x3d, self.data.y3d
            )

        self.assertTrue(
            "zentorch_matmul only supports Float and BFloat16"
            in str(context_int.exception)
        )

    def test_int_baddbmm_postop(self):
        self.data.create_data("int")
        with self.assertRaises(RuntimeError) as context_int:
            bias_as_postop = self.data.x3d.clone().to(torch.int)
            torch.ops.zentorch.zentorch_baddbmm(
                bias_as_postop, self.data.x3d, self.data.x3d
            )

        self.assertTrue(
            "zentorch_matmul only supports Float and BFloat16"
            in str(context_int.exception)
        )


if __name__ == "__main__":
    run_tests()
