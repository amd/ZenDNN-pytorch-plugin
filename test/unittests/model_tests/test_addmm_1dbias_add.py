# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import copy
import unittest
import torch
from parameterized import parameterized
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    counters,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
)


class Custom_Model_Addmm_1dbias_View_Add(nn.Module):
    def __init__(self, input_size, output_size, dtype, bias=True):
        super(Custom_Model_Addmm_1dbias_View_Add, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size, dtype=dtype, bias=bias)

    def forward(self, input, batch1):
        # Forward pass with mm and ReLU fused
        x = self.linear1(batch1)
        mm_res = x.view(input.size())
        add_res = torch.add(mm_res, input)
        return add_res


class Custom_Model_Addmm_1dbias_Add(nn.Module):
    def __init__(self, input_size, output_size, dtype, bias=True):
        super(Custom_Model_Addmm_1dbias_Add, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size, dtype=dtype, bias=bias)

    def forward(self, input, batch1):
        # Forward pass with mm and ReLU fused
        x = self.linear1(batch1)
        add_res = torch.add(x, input)
        return add_res


class Custom_Model_Addmm_1dbias_View_Add_Add(nn.Module):
    def __init__(self, input_size, output_size, dtype):
        super(Custom_Model_Addmm_1dbias_View_Add_Add, self).__init__()
        self.linear1 = nn.Linear(input_size, output_size, dtype=dtype)

    def forward(self, input, batch1):
        # Forward pass with mm and ReLU fused
        x = self.linear1(batch1)
        mm_res = x.view(input.size())
        add_res = torch.add(mm_res, input)
        add_res_2 = torch.add(add_res, input)
        return add_res_2


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Addmm_1dbias_Add_Model(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_1dbias_view_add_with_bias_model(self, dtype):
        self.data.create_data(dtype)
        model = Custom_Model_Addmm_1dbias_View_Add(
            40, 30, self.data.get_torch_type(dtype), bias=True
        ).eval()
        zentorch_model = copy.deepcopy(model)
        for inp in self.data.T1:
            for i in range(len(self.data.x1)):
                model_output = model(inp, self.data.x1[i])
                reset_dynamo()
                compiled_graph = torch.compile(zentorch_model, backend="zentorch")
                counters.clear()
                self.assertEqual(
                    counters["zentorch"]["pattern_matcher_addmm_1dbias_add"], 0
                )
                compiled_graph_output = compiled_graph(inp, self.data.x1[i])
                self.assertEqual(
                    counters["zentorch"]["pattern_matcher_addmm_1dbias_add"], 1
                )
                self.assertEqual(
                    model_output, compiled_graph_output, atol=1e-2, rtol=1e-2
                )

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_1dbias_view_add_without_bias_model(self, dtype):
        self.data.create_data(dtype)
        model = Custom_Model_Addmm_1dbias_View_Add(
            40, 30, self.data.get_torch_type(dtype), bias=False
        ).eval()
        zentorch_model = copy.deepcopy(model)
        for inp in self.data.T1:
            for i in range(len(self.data.x1)):
                model_output = model(inp, self.data.x1[i])
                reset_dynamo()
                compiled_graph = torch.compile(zentorch_model, backend="zentorch")
                counters.clear()
                self.assertEqual(counters["zentorch"]["pattern_matcher_mm_add"], 0)
                compiled_graph_output = compiled_graph(inp, self.data.x1[i])
                self.assertEqual(counters["zentorch"]["pattern_matcher_mm_add"], 1)
                self.assertEqual(
                    model_output, compiled_graph_output, atol=1e-2, rtol=1e-2
                )

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_3d_linear_3d_add_model(self, dtype):
        new_dtype = self.data.get_torch_type(dtype)
        arg_1 = torch.randn((2, 20, 30), dtype=new_dtype)
        arg_2 = torch.randn((2, 20, 40), dtype=new_dtype)
        model = Custom_Model_Addmm_1dbias_Add(
            40, 30, self.data.get_torch_type(dtype)
        ).eval()
        zentorch_model = copy.deepcopy(model)
        model_output = model(arg_1, arg_2)
        reset_dynamo()
        compiled_graph = torch.compile(zentorch_model, backend="zentorch")
        compiled_graph_output = compiled_graph(arg_1, arg_2)
        self.assertEqual(model_output, compiled_graph_output, atol=1e-2, rtol=1e-2)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_addmm_1dbias_view_add_add_model(self, dtype):
        self.data.create_data(dtype)
        model = Custom_Model_Addmm_1dbias_View_Add_Add(
            40, 30, self.data.get_torch_type(dtype)
        ).eval()
        zentorch_model = copy.deepcopy(model)
        for inp in self.data.T1:
            for i in range(len(self.data.x1)):
                model_output = model(inp, self.data.x1[i])
                reset_dynamo()
                compiled_graph = torch.compile(zentorch_model, backend="zentorch")
                counters.clear()
                self.assertEqual(
                    counters["zentorch"]["pattern_matcher_addmm_1dbias_add_add"], 0
                )
                compiled_graph_output = compiled_graph(inp, self.data.x1[i])
                self.assertEqual(
                    counters["zentorch"]["pattern_matcher_addmm_1dbias_add_add"], 1
                )
                self.assertEqual(
                    model_output, compiled_graph_output, atol=1e-2, rtol=1e-2
                )


if __name__ == "__main__":
    run_tests()
