# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

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


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Embedding_Bag_Sum_nodes(nn.Module):
    def __init__(self, num_embeddings):
        super(Custom_Model_Embedding_Bag_Sum_nodes, self).__init__()
        self.eb_bags_grp = [
            torch.nn.EmbeddingBag(num_embeddings, 3, mode="sum") for _ in range(10)
        ]

    def forward(self, eb_input, eb_offset):
        outputs_grp = [op(eb_input, eb_offset) for op in self.eb_bags_grp]

        outputs_grp[5] = torch.sum(outputs_grp[5], dim=1, keepdim=True)
        outputs_grp[6] = torch.sum(outputs_grp[6], dim=1, keepdim=True)

        output = torch.sum(torch.cat(outputs_grp, dim=1), dim=0)

        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Embedding_Sum_nodes(nn.Module):
    def __init__(self, num_embeddings):
        super(Custom_Model_Embedding_Sum_nodes, self).__init__()
        self.emebdding_grp = [torch.nn.Embedding(num_embeddings, 3) for _ in range(10)]

    def forward(self, inputs):
        outputs_grp = [op(inputs) for op in self.emebdding_grp]

        outputs_grp[3] = torch.sum(outputs_grp[3], dim=1, keepdim=True)
        outputs_grp[5] = torch.sum(outputs_grp[3], dim=1, keepdim=True)

        output = torch.sum(torch.cat(outputs_grp, dim=1), dim=0)

        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
# Testing revealed one of the corner cases where the common output node can
# have heterogeneous nodes like embedding1, embedding2, sum1, sum2, embedding3.
# To test the above scenario, the following testcases are added.
# Both the group ops are being tested here, with the heterogeneous op being sum
class Test_Group_Embeded_Ops_With_Sum_Ops_Model(Zentorch_TestCase):
    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_group_eb_with_sum_model(self, dtype, freeze_opt):
        self.data.create_data(dtype)

        indices = self.data.emb_input
        offsets = self.data.offsets

        model = Custom_Model_Embedding_Bag_Sum_nodes(self.data.R)

        native_output = model(indices, offsets)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(
            compiled_graph,
            (indices, offsets),
            freeze_opt
        )
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_group_embedding_with_sum_model(self, dtype, freeze_opt):
        self.data.create_data(dtype)
        indices = self.data.emb_input
        model = Custom_Model_Embedding_Sum_nodes(self.data.R)
        native_output = model(indices)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(
            compiled_graph,
            (indices),
            freeze_opt
        )
        self.assertEqual(native_output, compiled_output)


if __name__ == "__main__":
    run_tests()
