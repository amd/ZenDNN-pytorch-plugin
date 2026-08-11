# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    AddmmTestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    update_supported_dtypes,
    zentorch,
    freeze_opt,
    test_with_freeze_opt,
)

supported_dtypes = update_supported_dtypes(supported_dtypes, "zentorch_linear")


class Custom_Model_Linear_ReLU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Custom_Model_Linear_ReLU, self).__init__()

        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        return torch.relu(self.linear2(x))


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Linear_ReLU_Model(AddmmTestCase):
    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes, freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_linear_relu_model(self, dtype, freeze_opt):

        model = Custom_Model_Linear_ReLU(
            self.data.n, self.data.m, self.data.k
        ).eval()
        if dtype == "bfloat16":
            model = model.to(torch.bfloat16)
        elif dtype == "float16":
            model = model.to(torch.float16)
        model_output = model(self.data.input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = test_with_freeze_opt(
            compiled_graph, (self.data.input), freeze_opt
        )
        self.assertEqual(model_output, compiled_graph_output)


if __name__ == "__main__":
    run_tests()
