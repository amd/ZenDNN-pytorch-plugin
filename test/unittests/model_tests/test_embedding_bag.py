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
class Custom_Model_Embedding_Bag(nn.Module):
    def __init__(self, embedding_dim, output_dim, dtype=torch.float):
        super(Custom_Model_Embedding_Bag, self).__init__()
        self.embedding = nn.EmbeddingBag(10000, embedding_dim, dtype=dtype)
        self.intermediate = nn.Linear(embedding_dim, output_dim, dtype=dtype)
        self.output = nn.Linear(output_dim, 1, dtype=dtype)

    def forward(self, input):
        embed = self.embedding(input)
        intermediate = self.intermediate(embed)
        output = self.output(intermediate)
        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Embedding_Bag_Model(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_embedding_bag_compile_model(self, dtype):

        new_dtype = self.data.get_torch_type(dtype)
        model = Custom_Model_Embedding_Bag(100, 10, dtype=new_dtype)
        input = torch.randint(0, 10000, (1, 10))
        model_output = model(input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = compiled_graph(input)
        self.assertAlmostEqual(
            model_output.item(), compiled_graph_output.item(), places=6
        )


if __name__ == "__main__":
    run_tests()
