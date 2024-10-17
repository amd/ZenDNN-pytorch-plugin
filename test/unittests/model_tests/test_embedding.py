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
class Custom_Model_Embedding(nn.Module):
    def __init__(self, embedding_dim, dtype=torch.float):
        super(Custom_Model_Embedding, self).__init__()
        self.embedding = nn.Embedding(10000, embedding_dim, dtype=dtype)

    def forward(self, input):
        embed = self.embedding(input)
        return embed


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Embedding_Model(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_embedding_compile_model(self, dtype):
        new_dtype = self.data.get_torch_type(dtype)
        model = Custom_Model_Embedding(100, dtype=new_dtype)
        input = torch.randint(0, 10000, (10,))
        model_output = model(input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(input)
        self.assertEqual(model_output, compiled_graph_output)


if __name__ == "__main__":
    run_tests()
