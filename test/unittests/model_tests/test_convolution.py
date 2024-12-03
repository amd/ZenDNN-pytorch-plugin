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


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Convolution(nn.Module):
    def __init__(self):
        super(Custom_Model_Convolution, self).__init__()
        self.convolution = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)

    def forward(self, input):
        output = self.convolution(input)
        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Convolution_Model(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_convolution_model(self, dtype):
        self.data.create_data(dtype)
        model = Custom_Model_Convolution()
        if dtype == "bfloat16":
            model = model.to(torch.bfloat16)
        model_output = model(self.data.conv_input)
        reset_dynamo()
        zentorch_graph = torch.compile(model, backend="zentorch")
        zentorch_graph_output = zentorch_graph(self.data.conv_input)
        self.assertEqual(
            model_output,
            zentorch_graph_output,
        )


if __name__ == "__main__":
    run_tests()
