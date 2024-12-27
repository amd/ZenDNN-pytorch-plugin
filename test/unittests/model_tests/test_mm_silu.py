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
class Test_MM_Silu_Model(Zentorch_TestCase):
    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_mm_with_bias_silu_model(self, dtype, freeze_opt):
        self.data.create_unittest_data(dtype)
        model = nn.Sequential(nn.Linear(self.data.n, self.data.m, bias=True), nn.SiLU())
        if dtype == "bfloat16":
            model = model.bfloat16()
        model_output = model(self.data.input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = test_with_freeze_opt(
            compiled_graph,
            (self.data.input),
            freeze_opt
        )
        self.assertEqual(model_output, compiled_graph_output)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_mm_without_bias_silu_model(self, dtype, freeze_opt):
        self.data.create_unittest_data(dtype)
        model = nn.Sequential(
            nn.Linear(self.data.n, self.data.m, bias=False), nn.SiLU()
        )
        if dtype == "bfloat16":
            model = model.bfloat16()
        model_output = model(self.data.input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_graph_output = test_with_freeze_opt(
            compiled_graph,
            (self.data.input),
            freeze_opt
        )
        self.assertEqual(model_output, compiled_graph_output)


if __name__ == "__main__":
    run_tests()
