# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    MMTestCase,
    AddmmTestCase,
    has_zentorch,
    run_tests,
    reset_dynamo,
    supported_dtypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_MM_Tanh(MMTestCase):
    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_mm_tanh(self, dtype):
        self.data.create_unittest_data(dtype)
        native_output = torch.tanh(torch.mm(self.data.x, self.data.y))
        zentorch_output = torch.ops.zentorch.zentorch_mm_tanh(self.data.x, self.data.y)

        self.assertEqual(native_output, zentorch_output)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Addmm_Tanh(AddmmTestCase):
    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_addmm_tanh(self, dtype):
        self.data.create_unittest_data(dtype)
        bias = self.data.input.clone()
        native_output = torch.tanh(torch.addmm(bias, self.data.x, self.data.y))
        zentorch_output = torch.ops.zentorch.zentorch_addmm_tanh(
            bias, self.data.x, self.data.y
        )
        self.assertEqual(native_output, zentorch_output)

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_addmm_tanh_with_alpha_beta(self, dtype):
        self.data.create_unittest_data(dtype)
        bias = self.data.input.clone()
        alpha = 1.5
        beta = 0.5
        native_output = torch.tanh(
            torch.addmm(bias, self.data.x, self.data.y, alpha=alpha, beta=beta)
        )
        zentorch_output = torch.ops.zentorch.zentorch_addmm_tanh(
            bias, self.data.x, self.data.y, alpha=alpha, beta=beta
        )
        self.assertEqual(native_output, zentorch_output)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Addmm_1dbias_Tanh(AddmmTestCase):
    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes
    )
    @torch.inference_mode()
    def test_addmm_1dbias_tanh(self, dtype):
        new_dtype = self.data.get_torch_type(dtype)
        arg_0 = torch.randn((30), dtype=new_dtype)
        arg_1 = torch.randn((20, 40), dtype=new_dtype)
        arg_2 = torch.randn((30, 40), dtype=new_dtype)
        reset_dynamo()
        output_1 = torch.tanh(torch.addmm(arg_0, arg_1, arg_2.t()))
        output_2 = torch.ops.zentorch.zentorch_addmm_1dbias_tanh(
            arg_0, arg_1, arg_2.t()
        )
        self.assertEqual(output_1, output_2, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
