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
class Custom_Model_Embedding_Bag_Group(nn.Module):
    def __init__(self, num_embeddings):
        super(Custom_Model_Embedding_Bag_Group, self).__init__()
        self.eb_bags_grp_0 = [torch.nn.EmbeddingBag(num_embeddings, 3, mode="sum")] * 5
        self.eb_bags_grp_1 = [torch.nn.EmbeddingBag(num_embeddings, 3, mode="sum")] * 10
        self.eb_bags_grp_2 = [torch.nn.EmbeddingBag(num_embeddings, 3, mode="sum")] * 6

    def forward(self, eb_input, eb_offset):
        eb_outputs_grp_0 = [
            self.eb_bags_grp_0[i](eb_input, eb_offset) for i in range(5)
        ]
        concat_eb_tensors_0 = torch.cat(eb_outputs_grp_0)

        eb_outputs_grp_1 = [
            self.eb_bags_grp_1[i](eb_input, eb_offset) for i in range(10)
        ]
        concat_eb_tensors_1 = torch.cat(eb_outputs_grp_1)

        eb_outputs_grp_2 = [
            self.eb_bags_grp_2[i](eb_input, eb_offset) for i in range(6)
        ]
        concat_eb_tensors_2 = torch.cat(eb_outputs_grp_2)

        output = torch.cat(
            [concat_eb_tensors_0, concat_eb_tensors_1, concat_eb_tensors_2]
        )

        return output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Embedding_Bag_Group_Model(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_embedding_bag_group_model(self, dtype):
        self.data.create_data(dtype)
        model = Custom_Model_Embedding_Bag_Group(self.data.R)
        indices = self.data.emb_input
        offsets = self.data.offsets
        fx_g = make_fx(model)(indices, offsets)
        fx_g_output = fx_g(indices, offsets)
        fx_g_optimized = zentorch.optimize(fx_g)
        fx_g_optimized_output = fx_g_optimized(indices, offsets)
        self.assertEqual(fx_g_output, fx_g_optimized_output)
        target = torch.ops.zentorch.zentorch_horizontal_embedding_bag_group.default
        group_eb_count = 0
        for node in fx_g_optimized.graph.nodes:
            if isinstance(node.target, torch._ops.OpOverload) and node.target == target:
                group_eb_count += 1

        self.assertEqual(group_eb_count, 3)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_embedding_bag_group_compile_model(self, dtype):
        self.data.create_data(dtype)
        model = Custom_Model_Embedding_Bag_Group(self.data.R)
        indices = self.data.emb_input
        offset = self.data.offsets
        native_output = model(indices, offset)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = compiled_graph(indices, offset)
        self.assertEqual(native_output, compiled_output)


if __name__ == "__main__":
    run_tests()
