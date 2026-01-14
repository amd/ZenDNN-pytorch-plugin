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
    skip_test_pt_2_0,
    supported_dtypes,
    zentorch,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Addmm_Op(MMTestCase):
    @parameterized.expand(supported_dtypes)
    # Switching to Hypothesis exposes more issues, so the existing methods are retained.
    # Please refer ZENAI-1947 for details
    # @MMTestCase.hypothesis_params_mm_itr(
    #     dtype_list=supported_dtypes
    # )
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_addmm_variants(self, dtype):

        self.data.create_unittest_data(dtype)

        # TODO
        # Skip test for bfloat16 dtype
        # for shape of x = (5, 8), y = (8, 9), and input = (5, 9)
        # ZENAI-858
        if (
            self.data.x.size() == (5, 8)
            and self.data.y.size() == (8, 9)
            and self.data.input.size() == (5, 9)
        ):
            self.skipTest(
                "Skipping test for specific dimensions " "(5, 8), (8, 9), (5, 9)"
            )

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

        # <- Failure start from here ->

        # addmm with scalar input/bias
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input_scalar, self.data.x, self.data.y
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input_scalar, self.data.x, self.data.y
            ),
        )
        new_dtype = self.data.get_torch_type(dtype)
        # addmm with 2D input (1, n) -> For this, support can be added via per channel, binary add
        input_1_n = torch.randn((1, self.data.n), dtype=new_dtype)
        self.assertEqual(
            torch._C._VariableFunctions.addmm(input_1_n, self.data.x, self.data.y),
            torch.ops.zentorch.zentorch_addmm(input_1_n, self.data.x, self.data.y),
        )
        # addmm with 2D input (m, 1) -> -> For this, we don't know
        input_m_1 = torch.randn((self.data.m, 1), dtype=new_dtype)
        self.assertEqual(
            torch._C._VariableFunctions.addmm(input_m_1, self.data.x, self.data.y),
            torch.ops.zentorch.zentorch_addmm(input_m_1, self.data.x, self.data.y),
        )
        # addmm with 2D input (1, 1) -> For this, support can be added via per tensor, binary add
        input_1_1 = torch.randn((1, 1), dtype=new_dtype)
        self.assertEqual(
            torch._C._VariableFunctions.addmm(input_1_1, self.data.x, self.data.y),
            torch.ops.zentorch.zentorch_addmm(input_1_1, self.data.x, self.data.y),
        )

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_addmm_mismatched_dimensions(self, dtype):
        # The test will not fail when k == n
        # When K == N, Dimensions will be compatible even after reshaping
        if self.data.k != self.data.n:
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
                "Incompatible dimensions/shape for self tensor in addmm op"
                in str(context.exception)
            )
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zentorch_addmm(self.data.x, self.data.x, self.data.y)
            self.assertTrue(
                "Incompatible dimensions/shape for self tensor in addmm op"
                in str(context.exception)
            )

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=['int']
    )
    def test_addmm_unsupported_dtype(self, dtype):

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm(self.data.input, self.data.x, self.data.y)

        self.assertTrue(
            "zentorch_matmul only supports Float and BFloat16" in str(context.exception)
        )

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=['float32']
    )
    def test_float_addmm_bfloat16_postop(self):
        bias_as_postop = self.data.input.clone().to(torch.bfloat16)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm(bias_as_postop, self.data.x, self.data.y)

        self.assertTrue(
            "zentorch_matmul only supports" in str(context.exception)
        )

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=['float32']
    )
    def test_float_addmm_float_postop(self):
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input, self.data.x, self.data.y
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input, self.data.x, self.data.y
            ),
        )

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=['bfloat16']
    )
    def test_bfloat16_addmm_int_postop(self):
        self.skip_if_bfloat16_unsupported_hardware()
        bias_as_postop = self.data.input.clone().to(torch.int)
        with self.assertRaises(RuntimeError) as context_int:
            torch.ops.zentorch.zentorch_addmm(bias_as_postop, self.data.x, self.data.y)

        self.assertTrue(
            "zentorch_matmul only supports BFloat16 post ops"
            in str(context_int.exception)
        )

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=['bfloat16']
    )
    def test_bfloat16_addmm_bfloat16_postop(self):
        self.skip_if_bfloat16_unsupported_hardware()
        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input, self.data.x, self.data.y
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input, self.data.x, self.data.y
            ),
        )

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=['int']
    )
    def test_int_addmm_postop(self):
        bias_as_postop = self.data.input.clone().to(torch.int)
        with self.assertRaises(RuntimeError) as context_int:
            torch.ops.zentorch.zentorch_addmm(bias_as_postop, self.data.x, self.data.y)

        self.assertTrue(
            "zentorch_matmul only supports Float and BFloat16"
            in str(context_int.exception)
        )

    @parameterized.expand(supported_dtypes)
    # Switching to Hypothesis exposes more issues, so the existing methods are retained.
    # Please refer ZENAI-1948 for details
    # @MMTestCase.hypothesis_params_mm_itr(
    #     dtype_list=supported_dtypes
    # )
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_addmm_relu_with_kw(self, dtype):

        self.data.create_unittest_data(dtype)

        # TODO
        # Skip test for bfloat dtype
        # for shape of x = (5, 8), y = (8, 9), and input = (5, 9)
        # ZENAI-858
        if (
            self.data.x.size() == (5, 8)
            and self.data.y.size() == (8, 9)
            and self.data.input.size() == (5, 9)
        ):
            self.skipTest(
                "Skipping test for specific dimensions " "(5, 8), (8, 9), (5, 9)"
            )

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

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=supported_dtypes
    )
    def test_addmm_with_zero_alpha(self, dtype):

        self.assertEqual(
            torch._C._VariableFunctions.addmm(
                self.data.input, self.data.x, self.data.y, alpha=0.0
            ),
            torch.ops.zentorch.zentorch_addmm(
                self.data.input, self.data.x, self.data.y, alpha=0.0
            ),
        )

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=supported_dtypes
    )
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_addmm_relu_without_kw(self, dtype):

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
