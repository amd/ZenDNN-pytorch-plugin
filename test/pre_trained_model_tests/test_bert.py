# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import copy
import unittest
import torch
from parameterized import parameterized
from itertools import product
from transformers import BertModel
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from pre_trained_model_utils import (  # noqa: 402
    Zentorch_TestCase,
    TestCase,
    has_zentorch,
    run_tests,
    supported_dtypes,
    reset_dynamo,
    set_seed,
    zentorch,
    freeze_opt,
    test_with_freeze_opt,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Bert_Model(Zentorch_TestCase):
    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_bert_base_model(self, dtype, freeze_opt):
        self.skip_if_bfloat16_path_issue(dtype)
        self.data.create_pretrained_model_data(dtype)
        native_model = BertModel.from_pretrained("bert-large-uncased").eval()
        inductor_model = copy.deepcopy(native_model)
        zentorch_graph = torch.compile(native_model, backend="zentorch")
        zentorch_graph_output = test_with_freeze_opt(
            zentorch_graph,
            (self.data.input_tensor),
            freeze_opt
        )
        reset_dynamo()
        inductor_graph = torch.compile(inductor_model, backend="inductor")
        inductor_graph_output = inductor_graph(self.data.input_tensor)

        self.assertEqual(
            zentorch_graph_output, inductor_graph_output, atol=1e-2, rtol=1e-5
        )


if __name__ == "__main__":
    run_tests()
