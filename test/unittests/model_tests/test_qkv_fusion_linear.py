# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    AddmmTestCase,
    DataTypes,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    test_with_freeze_opt,
    counters,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_QKV_Linear(nn.Module):
    def __init__(self, dtype, input_dim, hidden_dim):
        super(Custom_Model_QKV_Linear, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Three parallel linear layers representing Query, Key, Value projections
        self.linear_q = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)
        self.linear_k = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)
        self.linear_v = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)

    def forward(self, x):
        # Q = X @ Wq
        q = self.linear_q(x)
        # K = X @ Wk
        k = self.linear_k(x)
        # V = X @ Wv
        v = self.linear_v(x)

        # Return concatenated for easy comparison
        return torch.cat([q, k, v], dim=-1)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_QKV_Fusion_Linear_Model(AddmmTestCase):

    def setUp(self):
        super().setUp()
        self._original_zentorch_linear = os.environ.get("ZENTORCH_LINEAR")
        os.environ["ZENTORCH_LINEAR"] = "1"

    def tearDown(self):
        if self._original_zentorch_linear is None:
            os.environ.pop("ZENTORCH_LINEAR", None)
        else:
            os.environ["ZENTORCH_LINEAR"] = self._original_zentorch_linear
        super().tearDown()

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes,
        freeze_list=[True],
    )
    @torch.inference_mode()
    def test_qkv_fusion_linear_model(self, dtype, freeze_opt=True):
        # Create model with QKV linear layers using parameterized dimensions
        # input_dim = k, hidden_dim = n from test data
        model = Custom_Model_QKV_Linear(
            dtype=DataTypes.get_torch_type(dtype),
            input_dim=self.data.k,
            hidden_dim=self.data.n,
        )
        # Create input tensor with shape (batch_size, seq_len, input_dim)
        # batch_size = b, seq_len = m, input_dim = k from test data
        input_tensor = torch.randn(
            self.data.b, self.data.m, self.data.k, dtype=DataTypes.get_torch_type(dtype)
        )

        native_output = model(input_tensor)
        reset_dynamo()
        counters.clear()
        self.assertEqual(counters["zentorch"]["qkv_fusion_linear"], 0)
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(
            compiled_graph, (input_tensor,), freeze_opt=True
        )
        self.assertEqual(counters["zentorch"]["qkv_fusion_linear"], 1)
        self.assertEqual(native_output, compiled_output)


if __name__ == "__main__":
    run_tests()
