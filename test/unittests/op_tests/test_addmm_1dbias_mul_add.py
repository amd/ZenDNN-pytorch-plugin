# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from parameterized import parameterized
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    AddmmTestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Addmm_1dbias_Mul_Add(AddmmTestCase):
    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_addmm_1dbias_mul_add_mismatched_dimensions(self, dtype):
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm_1dbias_mul_add(
                self.data.input1d,
                self.data.x,
                self.data.y,
                torch.reshape(
                    self.data.input,
                    (1, list(self.data.input.shape)[0], list(self.data.input.shape)[1]),
                ),
                torch.reshape(
                    self.data.input,
                    (1, list(self.data.input.shape)[0], list(self.data.input.shape)[1]),
                ),
            )
        self.assertTrue(
            "unsupported dims for mat1, mat2, "
            "binary1_input and binary2_input" in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    # Switching to Hypothesis exposes more issues, so the existing methods are retained.
    # Please refer ZENAI-1964 for details
    # @AddmmTestCase.hypothesis_params_addmm_itr(
    #     dtype_list=supported_dtypes
    # )
    @torch.inference_mode()
    def test_addmm_1dbias_mul_add_mismatched_sizes(self, dtype):
        self.data.create_unittest_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm_1dbias_mul_add(
                self.data.input1d, self.data.x, self.data.y, self.data.x, self.data.x
            )
        self.assertTrue(
            "unsupported sizes for mat1, mat2, "
            "binary1_input and binary2_input" in str(context.exception)
        )

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_addmm_1dbias_mul_add(self, dtype):
        self.skip_if_bfloat16_path_issue(dtype)
        new_dtype = self.data.get_torch_type(dtype)
        arg_0 = torch.rand((30), dtype=new_dtype)
        arg_1 = torch.rand((20, 40), dtype=new_dtype)
        arg_2 = torch.rand((30, 40), dtype=new_dtype)
        arg_3 = torch.rand((20, 30), dtype=new_dtype)
        reset_dynamo()
        output_1 = torch.add(
            torch.mul(torch.nn.functional.linear(arg_1, arg_2, arg_0), arg_3), arg_3
        )
        output_2 = torch.ops.zentorch.zentorch_addmm_1dbias_mul_add(
            arg_0, arg_1, arg_2.t(), arg_3, arg_3
        )
        self.assertEqual(output_1, output_2, atol=1e-9, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
