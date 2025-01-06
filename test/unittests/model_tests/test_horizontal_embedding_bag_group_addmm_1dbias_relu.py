# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from parameterized import parameterized
from itertools import product
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


class Custom_Model_Group_Embedding_Bag_Addmm_1dbias_Relu(torch.nn.Module):
    def __init__(self, num_embeddings, k):
        super(Custom_Model_Group_Embedding_Bag_Addmm_1dbias_Relu, self).__init__()
        # Common Nodes
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self.eb_bags = [torch.nn.EmbeddingBag(num_embeddings, 3)] * 2

        self.bmlp_0 = torch.nn.Linear(k, 4)
        self.bmlp_1 = torch.nn.Linear(4, 4)
        self.bmlp_2 = torch.nn.Linear(4, 3)

        self.tmlp_0 = torch.nn.Linear(12, 4)
        self.tmlp_1 = torch.nn.Linear(4, 2)
        self.tmlp_2 = torch.nn.Linear(2, 2)
        self.tmlp_3 = torch.nn.Linear(2, 1)

    def forward(self, eb_inputs, eb_offsets, mlp_inputs):

        outputs = []

        for _ in range(3):
            eb_outputs = [eb_op(eb_inputs, eb_offsets) for eb_op in self.eb_bags]

            mlp_outputs = self.bmlp_0(mlp_inputs)
            mlp_outputs = self.relu(mlp_outputs)
            mlp_outputs = self.bmlp_1(mlp_outputs)
            mlp_outputs = self.relu(mlp_outputs)
            mlp_outputs = self.bmlp_2(mlp_outputs)
            mlp_outputs = self.relu(mlp_outputs)

            interaction_input = eb_outputs + [mlp_outputs]
            interaction_output = torch.concat(interaction_input, dim=1)

            tmlp_input = torch.concat([mlp_outputs, interaction_output], dim=1)

            tmlp_outputs = self.tmlp_0(tmlp_input)
            tmlp_outputs = self.relu(tmlp_outputs)
            tmlp_outputs = self.tmlp_1(tmlp_outputs)
            tmlp_outputs = self.relu(tmlp_outputs)
            tmlp_outputs = self.tmlp_2(tmlp_outputs)
            tmlp_outputs = self.relu(tmlp_outputs)
            tmlp_outputs = self.tmlp_3(tmlp_outputs)
            tmlp_outputs = self.sigmoid(tmlp_outputs)

            outputs.append(tmlp_outputs)

        return outputs


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Group_Embedding_Bag_Addmm_1dbias_Relu_Model(Zentorch_TestCase):
    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_group_embedding_bag_addmm_1dbias_relu_model(self, dtype, freeze_opt):
        self.data.create_data(dtype)
        indices = self.data.emb_input
        offsets = self.data.offsets
        mlp_inputs = self.data.mlp_inputs
        model = Custom_Model_Group_Embedding_Bag_Addmm_1dbias_Relu(
            self.data.R, self.data.k
        )
        native_output = model(indices, offsets, mlp_inputs)
        reset_dynamo()
        compiled_model = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(
            compiled_model,
            (indices, offsets, mlp_inputs),
            freeze_opt
        )
        self.assertEqual(native_output, compiled_output)


if __name__ == "__main__":
    run_tests()
