# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
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
class Custom_Model_Embedding(nn.Module):
    def __init__(self, embedding_dim, dtype=torch.float):
        super(Custom_Model_Embedding, self).__init__()
        self.embedding = nn.Embedding(10000, embedding_dim, dtype=dtype)

    def forward(self, input):
        embed = self.embedding(input)
        return embed


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Embedding_Model(EmbTestCase):
    @EmbTestCase.hypothesis_params_emb_itr(
        dtype_list=supported_dtypes,
        freeze_list=freeze_opt
    )
    @torch.inference_mode()
    def test_embedding_compile_model(self, dtype, freeze_opt):
        new_dtype = self.data.get_torch_type(dtype)
        model = Custom_Model_Embedding(256, dtype=new_dtype)
        input = torch.randint(0, 10000, (10,))
        model_output = model(input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_embedding"], 0)
        compiled_graph_output = test_with_freeze_opt(
            compiled_graph,
            (input),
            freeze_opt
        )
        self.assertEqual(counters["zentorch"]["zentorch_embedding"], 1)
        self.assertEqual(model_output, compiled_graph_output)


if __name__ == "__main__":
    run_tests()
