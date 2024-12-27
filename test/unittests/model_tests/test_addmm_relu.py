# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import copy
import unittest
import torch
from parameterized import parameterized
from itertools import product
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    zentorch,
    freeze_opt,
    test_with_freeze_opt,
)


class Custom_Model_Addmm_Relu2(nn.Module):
    def __init__(self):
        super(Custom_Model_Addmm_Relu2, self).__init__()

    def forward(self, input, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        add_res = torch.add(mm_res, input)
        relu1_res = torch.relu(add_res)
        addmm_res = torch.addmm(relu1_res, batch1, batch2, beta=1.7, alpha=1.6)
        relu2_res = torch.relu(addmm_res)
        return relu2_res


class Custom_Model_Addmm_Relu1(nn.Module):
    def __init__(self, input_size, output_size):
        super(Custom_Model_Addmm_Relu1, self).__init__()

        # Linear layer (addmm operation)
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        # Forward pass with addmm and ReLU fused
        return torch.relu(self.linear(x))


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Addmm_Relu_Model(Zentorch_TestCase):
    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_addmm_relu2_model(self, dtype, freeze_opt):
        self.skip_if_bfloat16_path_issue(dtype)
        self.data.create_unittest_data(dtype)
        model = Custom_Model_Addmm_Relu2().eval()
        for inp in self.data.M:
            for i in range(len(self.data.x1)):
                for j in range(len(self.data.y1)):
                    model_output = model(inp, self.data.x1[i], self.data.y1[j])
                    reset_dynamo()
                    compiled_graph = torch.compile(model, backend="zentorch")
                    compiled_graph_output = test_with_freeze_opt(
                        compiled_graph,
                        (inp, self.data.x1[i], self.data.y1[j]),
                        freeze_opt
                    )
                    self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_addmm_relu1_model(self, dtype, freeze_opt):
        self.data.create_unittest_data(dtype)
        model = Custom_Model_Addmm_Relu1(self.data.n, self.data.m).eval()
        if dtype == "bfloat16":
            model = model.bfloat16()
        model_output = model(self.data.input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = test_with_freeze_opt(
            compiled_graph,
            (self.data.input),
            freeze_opt
        )
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    @unittest.skip("Nan and Inf giving non-deterministic output")
    def test_addmm_relu1_model_with_nan_or_inf(self, dtype, freeze_opt):
        if dtype == "bfloat16":
            self.skipTest("Skipping it since this testcase is not applicable for BF16.")

        self.data.create_unittest_data(dtype)
        model = Custom_Model_Addmm_Relu1(self.data.n, self.data.m).eval()
        # Nan's output is non-deterministic. Skipping Nan
        # self.data.input[0][0] = float("nan")
        self.data.input[1][1] = float("inf")
        reset_dynamo()
        zentorch_model = copy.deepcopy(model)
        inductor_graph = torch.compile(model, backend="inductor")
        inductor_graph_output = inductor_graph(self.data.input)
        reset_dynamo()
        zentorch_graph = torch.compile(zentorch_model, backend="zentorch")
        zentorch_graph_output = test_with_freeze_opt(
            zentorch_graph,
            (self.data.input),
            freeze_opt
        )
        self.assertEqual(inductor_graph_output, zentorch_graph_output)


if __name__ == "__main__":
    run_tests()
