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
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    skip_test_pt_2_0,
    supported_dtypes,
    zentorch,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Addmm_Op(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_addmm_variants(self, dtype):

        self.data.create_data(dtype)
        # addmm
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input, self.data.x, self.data.y
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input,
                self.data.x,
                self.data.y,
            ),
        )
        # addmm with kw_only arguments
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input, self.data.x, self.data.y, beta=1.3
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input, self.data.x, self.data.y, beta=1.3
            ),
        )

        # addmm with kw_only arguments
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input, self.data.x, self.data.y, alpha=1.3
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input, self.data.x, self.data.y, alpha=1.3
            ),
        )

        # addmm with kw_only arguments
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input, self.data.x, self.data.y, alpha=1.3, beta=1.3
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input, self.data.x, self.data.y, alpha=1.3, beta=1.3
            ),
        )

        # addmm with 1-d input
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input1d, self.data.x, self.data.y, alpha=1.3, beta=1.3
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input1d, self.data.x, self.data.y, alpha=1.3, beta=1.3
            ),
        )

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_mismatched_dimensions(self, dtype):
        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm(
                self.data.x,
                self.data.x,
                torch.reshape(
                    self.data.x,
                    (list(self.data.x.shape)[0], list(self.data.x.shape)[1], 1),
                ),
            )
        self.assertTrue(
            "unsupported dims for self, mat1 and mat2" in str(context.exception)
        )
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm(self.data.x3d, self.data.x, self.data.x)
            self.assertTrue(
                "unsupported dims for self, mat1 and mat2!" in str(context.exception)
            )

    @parameterized.expand(["int"])
    def test_addmm_unsupported_dtype(self, dtype):

        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm(self.data.input, self.data.x, self.data.y)

        self.assertTrue(
            "zentorch_matmul only supports Float and BFloat16" in str(context.exception)
        )

    def test_float_addmm_bfloat16_postop(self):
        self.data.create_data("float32")
        with self.assertRaises(RuntimeError) as context:
            bias_as_postop = self.data.input.clone().to(torch.bfloat16)
            torch.ops.zentorch.zentorch_addmm(bias_as_postop, self.data.x, self.data.y)

        self.assertTrue(
            "zentorch_matmul only supports Float and BFloat16" in str(context.exception)
        )

    def test_float_addmm_float_postop(self):
        self.data.create_data("float32")
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input, self.data.x, self.data.y
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input, self.data.x, self.data.y
            ),
        )

    def test_bfloat16_addmm_int_postop(self):
        self.skip_if_bfloat16_unsupported_hardware()
        self.data.create_data("bfloat16")
        with self.assertRaises(RuntimeError) as context_int:
            bias_as_postop = self.data.input.clone().to(torch.int)
            torch.ops.zentorch.zentorch_addmm(bias_as_postop, self.data.x, self.data.y)

        self.assertTrue(
            "zentorch_matmul only supports Float and BFloat16"
            in str(context_int.exception)
        )

    def test_bfloat16_addmm_bfloat16_postop(self):
        self.skip_if_bfloat16_unsupported_hardware()
        self.data.create_data("bfloat16")
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input, self.data.x, self.data.y
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input, self.data.x, self.data.y
            ),
        )

    def test_int_addmm_postop(self):
        self.data.create_data("int")
        with self.assertRaises(RuntimeError) as context_int:
            bias_as_postop = self.data.input.clone().to(torch.int)
            torch.ops.zentorch.zentorch_addmm(bias_as_postop, self.data.x, self.data.y)

        self.assertTrue(
            "zentorch_matmul only supports Float and BFloat16"
            in str(context_int.exception)
        )

    @parameterized.expand(supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_addmm_relu_with_kw(self, dtype):

        self.data.create_data(dtype)
        # addmm->relu
        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(
                    self.data.input, self.data.x, self.data.y, beta=1.5, alpha=1.7
                )
            ),
            torch.ops.zentorch.zentorch_addmm_relu(
                self.data.input, self.data.x, self.data.y, beta=1.5, alpha=1.7
            ),
        )

        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(
                    self.data.input, self.data.x, self.data.y, alpha=1.7
                )
            ),
            torch.ops.zentorch.zentorch_addmm_relu(
                self.data.input, self.data.x, self.data.y, alpha=1.7
            ),
        )

        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(
                    self.data.input, self.data.x, self.data.y, beta=1.5
                )
            ),
            torch.ops.zentorch.zentorch_addmm_relu(
                self.data.input, self.data.x, self.data.y, beta=1.5
            ),
        )

        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(
                    self.data.input, self.data.x, self.data.y, beta=0.0
                )
            ),
            torch.ops.zentorch.zentorch_addmm_relu(
                self.data.input, self.data.x, self.data.y, beta=0.0
            ),
        )

    @parameterized.expand(supported_dtypes)
    def test_addmm_with_zero_alpha(self, dtype):

        self.data.create_data(dtype)
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input, self.data.x, self.data.y, alpha=0.0
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input, self.data.x, self.data.y, alpha=0.0
            ),
        )

    @parameterized.expand(supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_addmm_relu_without_kw(self, dtype):

        self.data.create_data(dtype)
        # addmm->relu
        self.assertEqual(
            torch._C._VariableFunctions.relu(
                torch._C._VariableFunctions.addmm(
                    self.data.input, self.data.x, self.data.y
                )
            ),
            torch.ops.zentorch.zentorch_addmm_relu(
                self.data.input, self.data.x, self.data.y
            ),
        )


if __name__ == "__main__":
    run_tests()
