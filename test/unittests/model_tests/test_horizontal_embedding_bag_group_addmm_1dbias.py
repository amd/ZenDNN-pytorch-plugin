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
class Custom_Model_Group_Embedding_Bag_Addmm_1dbias(nn.Module):
    def __init__(self, num_embeddings, k):
        super(Custom_Model_Group_Embedding_Bag_Addmm_1dbias, self).__init__()
        self.eb_bags_grp = [torch.nn.EmbeddingBag(num_embeddings, 3)] * 3
        self.mlp_0 = torch.nn.Linear(k, 12)
        self.mlp_1 = torch.nn.Linear(12, 6)
        self.mlp_2 = torch.nn.Linear(6, 3)

    def forward(self, eb_input, eb_offset, mlp_input):
        eb_grp_outputs = [self.eb_bags_grp[i](eb_input, eb_offset) for i in range(3)]
        mlp_output = self.mlp_0(mlp_input)
        mlp_output = self.mlp_1(mlp_output)
        mlp_output = self.mlp_2(mlp_output)

        outputs = eb_grp_outputs + [mlp_output]
        outputs = torch.cat(outputs, dim=1)

        return outputs


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Group_Addmm_1dbias_Embedding_Bag(nn.Module):
    def __init__(self, num_embeddings, k):
        super(Custom_Model_Group_Addmm_1dbias_Embedding_Bag, self).__init__()
        self.eb_bags_grp = [torch.nn.EmbeddingBag(num_embeddings, 3)] * 3
        self.mlp_0 = torch.nn.Linear(k, 12)
        self.mlp_1 = torch.nn.Linear(12, 6)
        self.mlp_2 = torch.nn.Linear(6, 3)

    def forward(self, eb_input, eb_offset, mlp_input):
        mlp_output = self.mlp_0(mlp_input)
        mlp_output = self.mlp_1(mlp_output)
        mlp_output = self.mlp_2(mlp_output)

        eb_grp_outputs = [self.eb_bags_grp[i](eb_input, eb_offset) for i in range(3)]

        outputs = eb_grp_outputs + [mlp_output]
        outputs = torch.cat(outputs, dim=1)

        return outputs


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Group_Embedding_Bad_Addmm_1dbias_Model(Zentorch_TestCase):
    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_group_embedding_bag_addmm_1dbias_model(self, dtype, freeze_opt):
        self.data.create_data(dtype)
        indices = self.data.emb_input
        offsets = self.data.offsets
        mlp_inputs = self.data.mlp_inputs
        model = Custom_Model_Group_Embedding_Bag_Addmm_1dbias(self.data.R, self.data.k)
        native_output = model(indices, offsets, mlp_inputs)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(
            compiled_graph, (indices, offsets, mlp_inputs), freeze_opt
        )
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_group_addmm_1dbias_embedding_bag_model(self, dtype, freeze_opt):
        self.data.create_data(dtype)
        indices = self.data.emb_input
        offsets = self.data.offsets
        mlp_inputs = self.data.mlp_inputs
        model = Custom_Model_Group_Addmm_1dbias_Embedding_Bag(self.data.R, self.data.k)
        native_output = model(indices, offsets, mlp_inputs)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(
            compiled_graph, (indices, offsets, mlp_inputs), freeze_opt
        )
        self.assertEqual(native_output, compiled_output)


if __name__ == "__main__":
    run_tests()
