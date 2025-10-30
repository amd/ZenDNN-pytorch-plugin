# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import copy
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    EmbTestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    freeze_opt,
    test_with_freeze_opt,
    counters,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Single_Embedding(nn.Module):
    def __init__(self, num_embeddings):
        super(Custom_Model_Single_Embedding, self).__init__()
        self.embedding_1 = torch.nn.Embedding(num_embeddings, 3)

    def forward(self, inputs):
        output = self.embedding_1(inputs)
        return output


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
            torch.nn.EmbeddingBag(num_embeddings, 16, mode="sum"),
            torch.nn.Embedding(num_embeddings, 16),
            torch.nn.EmbeddingBag(num_embeddings, 16, mode="sum"),
            torch.nn.Embedding(num_embeddings, 16),
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
            torch.nn.EmbeddingBag(num_embeddings, 16, mode="sum"),
            torch.nn.Embedding(num_embeddings, 16),
            torch.nn.EmbeddingBag(num_embeddings, 16, mode="sum"),
            torch.nn.Embedding(num_embeddings, 16),
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
        self.e_bags_grp_0 = [torch.nn.Embedding(num_embeddings, 3) for _ in range(5)]
        self.e_bags_grp_1 = [torch.nn.Embedding(num_embeddings, 3) for _ in range(10)]
        self.e_bags_grp_2 = [torch.nn.Embedding(num_embeddings, 3) for _ in range(6)]

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
class Test_Embedding_Group_Model(EmbTestCase):
    @EmbTestCase.hypothesis_params_emb_itr(dtype_list=supported_dtypes)
    @torch.inference_mode()
    def test_embedding_group_model(self, dtype):
        model = Custom_Model_Embedding_Group(self.data.R)
        zentorch_model = copy.deepcopy(model)
        x = self.data.emb_input
        model = torch.compile(model, backend="inductor")
        model_output = model(x)
        reset_dynamo()
        compiled_model = torch.compile(zentorch_model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_embedding"], 0)
        self.assertEqual(counters["zentorch"]["zentorch_horizontal_embedding_group"], 0)
        compiled_model_output = compiled_model(x)
        self.assertEqual(counters["zentorch"]["zentorch_embedding"], 21)
        self.assertEqual(counters["zentorch"]["zentorch_horizontal_embedding_group"], 3)
        self.assertEqual(model_output, compiled_model_output)

    @EmbTestCase.hypothesis_params_emb_itr(
        dtype_list=supported_dtypes, freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_embedding_group_compile_model(self, dtype, freeze_opt):
        model = Custom_Model_Embedding_Group(self.data.R)
        x = self.data.emb_input
        native_output = model(x)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(compiled_graph, (x), freeze_opt)
        self.assertEqual(native_output, compiled_output)

    @EmbTestCase.hypothesis_params_emb_itr(
        dtype_list=supported_dtypes, freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_emb_emb_bag_common_node_model(self, dtype, freeze_opt):
        model = Custom_Model_Emb_Emb_Bag_Common_Node(self.data.R)
        indices = self.data.emb_input
        offsets = self.data.offsets
        native_output = model(indices, offsets)
        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_horizontal_embedding_group"], 0)
        self.assertEqual(
            counters["zentorch"]["zentorch_horizontal_embedding_bag_group"], 0
        )
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(
            compiled_graph, (indices, offsets), freeze_opt
        )
        self.assertEqual(counters["zentorch"]["zentorch_horizontal_embedding_group"], 1)
        self.assertEqual(
            counters["zentorch"]["zentorch_horizontal_embedding_bag_group"], 1
        )
        self.assertEqual(native_output, compiled_output)

    @EmbTestCase.hypothesis_params_emb_itr(
        dtype_list=supported_dtypes, freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_emb_emb_bag_diff_node_model(self, dtype, freeze_opt):
        model = Custom_Model_Emb_Emb_Bag_Diff_Node(self.data.R)
        indices = self.data.emb_input
        offsets = self.data.offsets
        native_output = model(indices, offsets)
        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_horizontal_embedding_group"], 0)
        self.assertEqual(
            counters["zentorch"]["zentorch_horizontal_embedding_bag_group"], 0
        )
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(
            compiled_graph, (indices, offsets), freeze_opt
        )
        self.assertEqual(counters["zentorch"]["zentorch_horizontal_embedding_group"], 1)
        self.assertEqual(
            counters["zentorch"]["zentorch_horizontal_embedding_bag_group"], 1
        )
        self.assertEqual(native_output, compiled_output)

    @EmbTestCase.hypothesis_params_emb_itr(
        dtype_list=supported_dtypes, freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_embedding_model(self, dtype, freeze_opt):
        model = Custom_Model_Embedding(self.data.R)
        indices = torch.cat([torch.unsqueeze(self.data.emb_input, dim=0)] * 2)
        model = torch.compile(model, backend="inductor")
        native_output = model(indices)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(compiled_graph, (indices), freeze_opt)
        self.assertEqual(native_output, compiled_output)

    @EmbTestCase.hypothesis_params_emb_itr(
        dtype_list=supported_dtypes, freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_single_embedding_compile_model(self, dtype, freeze_opt):
        model = Custom_Model_Single_Embedding(self.data.R)
        zentorch_model = copy.deepcopy(model)
        x = self.data.emb_input
        model = torch.compile(model, backend="inductor")
        native_output = model(x)
        reset_dynamo()
        counters.clear()
        compiled_graph = torch.compile(zentorch_model, backend="zentorch")
        compiled_output = test_with_freeze_opt(compiled_graph, (x), freeze_opt)
        self.assertEqual(counters["zentorch"]["zentorch_embedding"], 1)
        self.assertEqual(counters["zentorch"]["zentorch_horizontal_embedding_group"], 0)
        self.assertEqual(native_output, compiled_output)


if __name__ == "__main__":
    run_tests()
