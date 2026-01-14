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
    reset_dynamo,
    run_tests,
    supported_dtypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Addmm_1dbias_Add(AddmmTestCase):
    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_addmm_1dbias_add(self, dtype):
        new_dtype = self.data.get_torch_type(dtype)
        arg_0 = torch.randn((30), dtype=new_dtype)
        arg_1 = torch.randn((20, 40), dtype=new_dtype)
        arg_2 = torch.randn((30, 40), dtype=new_dtype)
        arg_3 = torch.randn((20, 30), dtype=new_dtype)
        reset_dynamo()
        output_1 = torch.add(torch.nn.functional.linear(arg_1, arg_2, arg_0), arg_3)
        output_2 = torch.ops.zentorch.zentorch_addmm_1dbias_add(
            arg_0, arg_1, arg_2.t(), arg_3
        )
        self.assertEqual(output_1, output_2, atol=1e-2, rtol=1e-2)

    # Disabling this test case as mixed precision is not supported currently
    @unittest.skipIf(True, "ZENTORCH currently doesn't support mixed precision")
    @torch.inference_mode()
    def test_addmm_1dbias_add_mp(self, dtype):
        # new_dtype = self.data.get_torch_type(dtype)
        arg_0 = torch.randn((30), dtype=torch.bfloat16)
        arg_1 = torch.randn((20, 40), dtype=torch.bfloat16)
        arg_2 = torch.randn((30, 40), dtype=torch.bfloat16)
        arg_3 = torch.randn((20, 30), dtype=torch.float32)
        reset_dynamo()
        output_1 = torch.add(torch.nn.functional.linear(arg_1, arg_2, arg_0), arg_3)
        output_2 = torch.ops.zentorch.zentorch_addmm_1dbias_add(
            arg_0, arg_1, arg_2.t(), arg_3
        )
        self.assertEqual(output_1, output_2, atol=1e-2, rtol=1e-2)

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_addmm_1dbias_add_mismatched_dimensions(self, dtype):
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm_1dbias_add(
                self.data.input1d,
                self.data.x,
                self.data.y,
                torch.reshape(
                    self.data.input,
                    (1, list(self.data.input.shape)[0], list(self.data.input.shape)[1]),
                ),
            )
        self.assertTrue(
            "unsupported dims for mat1, mat2 and post op buffers"
            in str(context.exception)
        )

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_addmm_1dbias_add_mismatched_sizes(self, dtype):
        # The test will not fail when k == n
        # When K == N, Dimensions will be compatible even after reshaping
        if self.data.k != self.data.n:
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zentorch_addmm_1dbias_add(
                    self.data.input1d, self.data.x, self.data.y, self.data.x
                )
            self.assertTrue(
                "unsupported shapes for mat1, mat2 and post op buffers"
                in str(context.exception)
            )

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_addmm_1dbias_add_add_mismatched_dimensions(self, dtype):
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_addmm_1dbias_add_add(
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
            "binary1_input and binary2_input"
            in str(context.exception)
        )

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_addmm_1dbias_add_add_mismatched_sizes(self, dtype):
        # The test will not fail when k == n
        # When K == N, Dimensions will be compatible even after reshaping
        if self.data.k != self.data.n:
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zentorch_addmm_1dbias_add_add(
                    self.data.input1d, self.data.x, self.data.y, self.data.x, self.data.x
                )
            self.assertTrue(
                "unsupported sizes for mat1, mat2, "
                "binary1_input and binary2_input"
                in str(context.exception)
            )

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_addmm_1dbias_add_add_op_level(self, dtype):
        new_dtype = self.data.get_torch_type(dtype)
        arg_0 = torch.rand((30), dtype=new_dtype)
        arg_1 = torch.rand((20, 40), dtype=new_dtype)
        arg_2 = torch.rand((30, 40), dtype=new_dtype)
        arg_3 = torch.rand((20, 30), dtype=new_dtype)
        reset_dynamo()
        output_1 = torch.add(
            torch.add(torch.nn.functional.linear(arg_1, arg_2, arg_0), arg_3), arg_3
        )
        output_2 = torch.ops.zentorch.zentorch_addmm_1dbias_add_add(
            arg_0, arg_1, arg_2.t(), arg_3, arg_3
        )
        self.assertEqual(output_1, output_2, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
