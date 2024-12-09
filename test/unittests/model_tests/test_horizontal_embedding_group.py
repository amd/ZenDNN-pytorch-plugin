# ******************************************************************************
# Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from parameterized import parameterized
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx
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
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Embedding(nn.Module):
    def __init__(self, num_embeddings):
        super(Custom_Model_Embedding, self).__init__()
        self.embedding_1 = torch.nn.Embedding(num_embeddings, 3)
        self.embedding_2 = torch.nn.Embedding(num_embeddings, 3)

    def forward(self, inputs):
        output = self.embedding_1(inputs) + self.embedding_2(inputs)

        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Emb_Emb_Bag_Diff_Node(nn.Module):
    def __init__(self, num_embeddings):
        super(Custom_Model_Emb_Emb_Bag_Diff_Node, self).__init__()
        self.eb_bags_grp = [
            torch.nn.EmbeddingBag(num_embeddings, 3, mode="sum"),
            torch.nn.Embedding(num_embeddings, 3),
            torch.nn.EmbeddingBag(num_embeddings, 3, mode="sum"),
            torch.nn.Embedding(num_embeddings, 3),
        ]

    def forward(self, eb_input, eb_offset):
        outputs_grp_0 = [
            self.eb_bags_grp[0](eb_input, eb_offset),
            self.eb_bags_grp[2](eb_input, eb_offset),
        ]
        outputs_grp_1 = [
            self.eb_bags_grp[1](eb_input),
            self.eb_bags_grp[3](eb_input),
        ]

        output_0 = torch.sum(torch.cat(outputs_grp_0), dim=0)
        output_1 = torch.sum(torch.cat(outputs_grp_1), dim=0)

        return torch.cat([output_0, output_1])


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Emb_Emb_Bag_Common_Node(nn.Module):
    def __init__(self, num_embeddings):
        super(Custom_Model_Emb_Emb_Bag_Common_Node, self).__init__()
        self.eb_bags_grp = [
            torch.nn.EmbeddingBag(num_embeddings, 3, mode="sum"),
            torch.nn.Embedding(num_embeddings, 3),
            torch.nn.EmbeddingBag(num_embeddings, 3, mode="sum"),
            torch.nn.Embedding(num_embeddings, 3),
        ]

    def forward(self, eb_input, eb_offset):
        outputs_grp = [
            self.eb_bags_grp[0](eb_input, eb_offset),
            self.eb_bags_grp[1](eb_input),
            self.eb_bags_grp[2](eb_input, eb_offset),
            self.eb_bags_grp[3](eb_input),
        ]

        output = torch.sum(torch.cat(outputs_grp), dim=0)

        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Embedding_Group(nn.Module):
    def __init__(self, num_embeddings):
        super(Custom_Model_Embedding_Group, self).__init__()
        self.e_bags_grp_0 = [torch.nn.Embedding(num_embeddings, 3)] * 5
        self.e_bags_grp_1 = [torch.nn.Embedding(num_embeddings, 3)] * 10
        self.e_bags_grp_2 = [torch.nn.Embedding(num_embeddings, 3)] * 6

    def forward(self, e_input):
        e_outputs_grp_0 = [self.e_bags_grp_0[i](e_input) for i in range(5)]
        e_sum_0 = torch.unsqueeze(torch.sum(torch.cat(e_outputs_grp_0), dim=0), dim=0)

        e_outputs_grp_1 = [self.e_bags_grp_1[i](e_input) for i in range(10)]
        e_sum_1 = torch.unsqueeze(torch.sum(torch.cat(e_outputs_grp_1), dim=0), dim=0)

        e_outputs_grp_2 = [self.e_bags_grp_2[i](e_input) for i in range(6)]
        e_sum_2 = torch.unsqueeze(torch.sum(torch.cat(e_outputs_grp_2), dim=0), dim=0)

        output = torch.cat([e_sum_0, e_sum_1, e_sum_2])

        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Embedding_Group_Model(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_embedding_group_model(self, dtype):
        self.data.create_data(dtype)
        model = Custom_Model_Embedding_Group(self.data.R)
        x = self.data.emb_input
        fx_g = make_fx(model)(x)
        fx_g_output = fx_g(x)
        fx_g_optimized = zentorch.optimize(fx_g)
        fx_g_optimized_output = fx_g_optimized(x)
        self.assertEqual(fx_g_output, fx_g_optimized_output)
        target = torch.ops.zentorch.zentorch_horizontal_embedding_group.default
        group_eb_count = 0
        for node in fx_g_optimized.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload) and node.target == target:
                group_eb_count += 1
        self.assertEqual(group_eb_count, 3)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_embedding_group_compile_model(self, dtype):
        self.data.create_data(dtype)
        model = Custom_Model_Embedding_Group(self.data.R)
        x = self.data.emb_input
        native_output = model(x)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = compiled_graph(x)
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_emb_emb_bag_common_node_model(self, dtype):
        self.data.create_data(dtype)
        model = Custom_Model_Emb_Emb_Bag_Common_Node(self.data.R)
        indices = self.data.emb_input
        offsets = self.data.offsets
        native_output = model(indices, offsets)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = compiled_graph(indices, offsets)
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_emb_emb_bag_diff_node_model(self, dtype):
        self.data.create_data(dtype)
        model = Custom_Model_Emb_Emb_Bag_Diff_Node(self.data.R)
        indices = self.data.emb_input
        offsets = self.data.offsets
        native_output = model(indices, offsets)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = compiled_graph(indices, offsets)
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_embedding_model(self, dtype):
        self.data.create_data(dtype)
        model = Custom_Model_Embedding(self.data.R)
        indices = torch.cat([torch.unsqueeze(self.data.emb_input, dim=0)] * 2)
        native_output = model(indices)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = compiled_graph(indices)
        self.assertEqual(native_output, compiled_output)


if __name__ == "__main__":
    run_tests()
