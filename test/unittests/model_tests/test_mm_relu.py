# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from parameterized import parameterized
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
)


class Custom_Model_MM_Relu2(nn.Module):
    def __init__(self):
        super(Custom_Model_MM_Relu2, self).__init__()

    def forward(self, batch1, batch2):
        mm_res = torch.mm(batch1, batch2)
        relu_res = torch.relu(mm_res)
        return relu_res


class Custom_Model_MM_ReLU1(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Custom_Model_MM_ReLU1, self).__init__()

        # Linear layers (mm operation)
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Forward pass with mm and ReLU fused
        x = torch.relu(self.linear1(x))
        return torch.relu(self.linear2(x))


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_MM_RELU_Model(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_mm_relu2_optimize_model(self, dtype):

        self.data.create_data(dtype)
        model = Custom_Model_MM_Relu2().eval()
        for i in range(len(self.data.x1)):
            for j in range(len(self.data.y1)):
                model_output = model(self.data.x1[i], self.data.y1[j])
                reset_dynamo()
                compiled_graph = torch.compile(model, backend="zentorch")
                compiled_graph_output = compiled_graph(self.data.x1[i], self.data.y1[j])
                self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_mm_relu2_zero_input_optimize_model(self, dtype):

        self.data.create_data(dtype)
        model = Custom_Model_MM_Relu2().eval()
        model_output = model(self.data.x1[0] * 0, self.data.y1[0] * 0)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(self.data.x1[0] * 0, self.data.y1[0] * 0)
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_mm_relu2_negative_input_optimize_model(self, dtype):

        self.data.create_data(dtype)
        model = Custom_Model_MM_Relu2().eval()
        model_output = model(self.data.x1[0] * -1, self.data.y1[0] * -1)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(
            self.data.x1[0] * -1, self.data.y1[0] * -1
        )
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_mm_relu1_model(self, dtype):

        self.data.create_data(dtype)
        model = Custom_Model_MM_ReLU1(self.data.n, self.data.m, self.data.k).eval()
        if dtype == "bfloat16":
            model = model.bfloat16()
        model_output = model(self.data.input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(self.data.input)
        self.assertEqual(model_output, compiled_graph_output)


if __name__ == "__main__":
    run_tests()
