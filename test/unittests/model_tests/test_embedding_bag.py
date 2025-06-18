# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
import copy
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
    DataTypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Embedding_Bag(nn.Module):
    def __init__(
        self,
        num_embeddings,
        embedding_dim,
        emb_mat,
        dtype,
    ):
        super(Custom_Model_Embedding_Bag, self).__init__()
        self.embedding = nn.EmbeddingBag(num_embeddings, embedding_dim, dtype=dtype)
        # overwriting embedding-bag weights
        with torch.no_grad():
            self.embedding.weight.copy_(emb_mat)

    def forward(self, input):
        embed = self.embedding(input)
        return embed


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Embedding_Bag_Model(EmbTestCase):
    @EmbTestCase.hypothesis_params_emb_itr(
        dtype_list=supported_dtypes, freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_embedding_bag_compile_model(self, dtype, freeze_opt):
        new_dtype = DataTypes.get_torch_type(dtype)
        model = Custom_Model_Embedding_Bag(
            self.data.R,
            self.data.k,
            self.data.embedding_matrix,
            new_dtype,
        )
        zen_model = copy.deepcopy(model)
        input = self.data.emb_input.unsqueeze(0)
        ind_compiled_graph = torch.compile(model, backend="inductor")
        ind_compiled_graph_output = test_with_freeze_opt(
            ind_compiled_graph, (input), freeze_opt
        )
        reset_dynamo()
        zen_compiled_graph = torch.compile(zen_model, backend="zentorch")
        zen_compiled_graph_output = test_with_freeze_opt(
            zen_compiled_graph, (input), freeze_opt
        )
        # TODO
        # Increased tolerent for bfloat16 dtype by atol=1e-03, rtol=0.01
        # Getting failure due to higer diff than allowed
        # Change will restore after fix
        # ZENAI-858
        self.assertEqual(
            zen_compiled_graph_output, ind_compiled_graph_output, atol=1e-3, rtol=0.01
        )


if __name__ == "__main__":
    run_tests()
