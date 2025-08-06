# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import copy
import unittest
import torch
from parameterized import parameterized
from torch import nn
from itertools import product
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    AddmmTestCase,
    counters,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    skip_test_pt_2_1,
    freeze_opt,
    test_with_freeze_opt,
)


class Custom_Model_Addmm_1dbias_View_Mul_Add(nn.Module):
    def __init__(self, input_size, output_size, dtype):
        super(Custom_Model_Addmm_1dbias_View_Mul_Add, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size, dtype=dtype)

    def forward(self, input, batch1):
        # Forward pass with mm and ReLU fused
        x = self.linear1(batch1)
        mm_res = x.view(input.size())
        mul_res = torch.mul(mm_res, input)
        add_res = torch.add(mul_res, input)
        return add_res


class Custom_Model_Addmm_1dbias_Alpha_Beta_View_Mul_Add(nn.Module):
    def __init__(self):
        super(Custom_Model_Addmm_1dbias_Alpha_Beta_View_Mul_Add, self).__init__()

    def forward(self, mat1, mat2, bias, new_shape, add_input):
        addmm_result = torch.addmm(bias, mat1, mat2, alpha=1, beta=1.2)
        view_result = addmm_result.view(*new_shape)
        mul_result = view_result * add_input
        final_result = mul_result + add_input
        return final_result


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
@unittest.skipIf(skip_test_pt_2_1, "Pattern matcher disabled for Torch < 2.2")
class Test_Addmm_1dbias_Mul_Add_Model(AddmmTestCase):
    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes,
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_addmm_1dbias_view_mul_add_model(self, dtype, freeze_opt):
        self.skip_if_bfloat16_path_issue(dtype)
        model = Custom_Model_Addmm_1dbias_View_Mul_Add(
            40, 30, self.data.get_torch_type(dtype)
        ).eval()
        zentorch_model = copy.deepcopy(model)
        for inp in self.data.T1:
            for i in range(len(self.data.x1)):
                model_output = model(inp, self.data.x1[i])
                reset_dynamo()
                compiled_model = torch.compile(zentorch_model, backend="zentorch")
                counters.clear()
                self.assertEqual(
                    counters["zentorch"]["pattern_matcher_addmm_1dbias_mul_add"], 0
                )
                compiled_graph_output = test_with_freeze_opt(
                    compiled_model, (inp, self.data.x1[i]), freeze_opt
                )
                self.assertEqual(
                    counters["zentorch"]["pattern_matcher_addmm_1dbias_mul_add"], 1
                )
                self.assertEqual(
                    model_output, compiled_graph_output, atol=1e-2, rtol=1e-2
                )

    # Switching to Hypothesis exposes more issues, so the existing methods are retained.
    # Please refer ZENAI-1966 for details
    # @AddmmTestCase.hypothesis_params_addmm_itr(
    #     dtype_list=supported_dtypes,
    #     freeze_list=freeze_opt
    # )
    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_addmm_1dbias_alpha_beta_view_add_add_model(self, dtype, freeze_opt):
        self.skip_if_bfloat16_path_issue(dtype)
        self.data.create_unittest_data(dtype)
        test_dtype = self.data.get_torch_type(dtype)
        new_shape = (1, self.data.n, self.data.m)
        bias = self.data.input1d
        mat2 = self.data.y
        mat1 = self.data.x
        add_input = torch.randn(*new_shape, dtype=test_dtype)

        model = Custom_Model_Addmm_1dbias_Alpha_Beta_View_Mul_Add().eval()
        zentorch_model = copy.deepcopy(model)
        model_output = model(mat1, mat2, bias, new_shape, add_input)
        reset_dynamo()
        compiled_model = torch.compile(zentorch_model, backend="zentorch")
        counters.clear()
        self.assertEqual(
            counters["zentorch"]["pattern_matcher_addmm_1dbias_mul_add"], 0
        )
        compiled_graph_output = test_with_freeze_opt(
            compiled_model, (mat1, mat2, bias, new_shape, add_input), freeze_opt
        )
        self.assertEqual(
            counters["zentorch"]["pattern_matcher_addmm_1dbias_mul_add"], 1
        )
        self.assertEqual(model_output, compiled_graph_output, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
