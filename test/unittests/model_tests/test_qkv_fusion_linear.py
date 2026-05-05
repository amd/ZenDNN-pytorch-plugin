# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
import os
import copy
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
class Custom_Model_QKV_Linear_4(nn.Module):
    def __init__(self, dtype, input_dim, hidden_dim):
        super(Custom_Model_QKV_Linear_4, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.linear_1 = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)
        self.linear_2 = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)
        self.linear_3 = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)
        self.linear_4 = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)

    def forward(self, x):
        l1 = self.linear_1(x)
        l2 = self.linear_2(x)
        l3 = self.linear_3(x)
        l4 = self.linear_4(x)

        # Return concatenated for easy comparison
        return torch.cat([l1, l2, l3, l4], dim=-1)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_QKV_Linear_Longformer(nn.Module):
    """
    Matches Longformer's pattern:
    - Input is permuted before linear (simulating transformer hidden states)
    - Q goes through div, view, permute, view, as_strided, then to bmm
    - K goes through view, permute, view, as_strided, then to bmm
    - V is returned directly
    """

    def __init__(self, dtype, hidden_size=768, num_heads=12):
        super(Custom_Model_QKV_Linear_Longformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, hidden_size, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, hidden_size, dtype=dtype)

        self.scale = self.head_dim**0.5

    def forward(self, x):
        # x: [batch, seq_len, hidden] but we permute like Longformer does
        # Simulating: permute = [1, 0, 2] on input
        x_permuted = x.permute(1, 0, 2)  # [seq_len, batch, hidden]
        q = self.q_proj(x_permuted)
        k = self.k_proj(x_permuted)
        v = self.v_proj(x_permuted)

        seq_len, batch_size, _ = q.shape

        # Q path: view -> permute -> view -> as_strided -> bmm
        q = q / self.scale
        q = q.view(seq_len, batch_size, self.num_heads, self.head_dim)
        q = q.permute(1, 0, 2, 3)  # [batch, seq, heads, head_dim]
        q = q.permute(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
        q = q.reshape(self.num_heads, batch_size, seq_len, self.head_dim)
        q = q.reshape(self.num_heads, batch_size * seq_len, self.head_dim)
        q = q.as_strided(
            (self.num_heads, batch_size * seq_len, self.head_dim),
            (self.head_dim, self.num_heads * self.head_dim, 1),
        )

        # K path: view -> permute -> view -> as_strided -> transpose for bmm
        k = k.view(seq_len, batch_size, self.num_heads, self.head_dim)
        k = k.permute(1, 0, 2, 3)  # [batch, seq, heads, head_dim]
        k = k.permute(0, 2, 1, 3)  # [batch, heads, seq, head_dim]
        k = k.reshape(self.num_heads, batch_size, seq_len, self.head_dim)
        k = k.reshape(self.num_heads, batch_size * seq_len, self.head_dim)
        k = k.as_strided(
            (self.num_heads, batch_size * seq_len, self.head_dim),
            (self.head_dim, self.num_heads * self.head_dim, 1),
        )
        k = k.permute(0, 2, 1)  # [heads, head_dim, seq]

        # BMM: Q @ K^T
        attn = torch.bmm(q, k)  # [heads, seq, seq]

        return attn, v


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_QKV_Linear_ZenTorch(nn.Module):
    """QKV model calling zentorch_linear_unary directly with configurable kwargs."""

    def __init__(self, dtype, input_dim, hidden_dim,
                 is_weight_prepacked=False, post_ops=("none", "none", "none")):
        super().__init__()
        self.linear_q = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)
        self.linear_k = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)
        self.linear_v = nn.Linear(input_dim, hidden_dim, bias=True, dtype=dtype)
        self.is_weight_prepacked = is_weight_prepacked
        self.post_op_q = post_ops[0]
        self.post_op_k = post_ops[1]
        self.post_op_v = post_ops[2]

    def forward(self, x):
        q = torch.ops.zentorch.zentorch_linear_unary(
            x, self.linear_q.weight, self.linear_q.bias,
            is_weight_prepacked=self.is_weight_prepacked,
            post_op=self.post_op_q,
            zentorch_op_name="zentorch::zentorch_linear_unary",
        )
        k = torch.ops.zentorch.zentorch_linear_unary(
            x, self.linear_k.weight, self.linear_k.bias,
            is_weight_prepacked=self.is_weight_prepacked,
            post_op=self.post_op_k,
            zentorch_op_name="zentorch::zentorch_linear_unary",
        )
        v = torch.ops.zentorch.zentorch_linear_unary(
            x, self.linear_v.weight, self.linear_v.bias,
            is_weight_prepacked=self.is_weight_prepacked,
            post_op=self.post_op_v,
            zentorch_op_name="zentorch::zentorch_linear_unary",
        )
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

    # Added higher time_out for this test as it was failing with deadline exceeded error frequently
    # TODO: Investigate why it requires higher deadline: ZENAI-3638
    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes,
        freeze_list=[True],
        time_out=25000,
    )
    @torch.inference_mode()
    def test_qkv_fusion_linear_4_model(self, dtype, freeze_opt=True):
        # Create model with 4 linear layers using parameterized dimensions
        # input_dim = k, hidden_dim = n from test data
        model = Custom_Model_QKV_Linear_4(
            dtype=DataTypes.get_torch_type(dtype),
            input_dim=self.data.k,
            hidden_dim=self.data.n,
        )
        # Create input tensor with shape (batch_size, seq_len, input_dim)
        # batch_size = b, seq_len = m, input_dim = k from test data
        input_tensor = torch.randn(
            self.data.b, self.data.m, self.data.k, dtype=DataTypes.get_torch_type(dtype)
        )
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        with self.assertLogs("zentorch", level="INFO") as cm:
            test_with_freeze_opt(compiled_graph, (input_tensor,), freeze_opt=True)
        self.assertTrue(
            any(
                "Fusion only supported for exactly 3 linear nodes currently, skipping fusion"
                in message
                for message in cm.output
            )
        )

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes,
        freeze_list=[True],
    )
    @torch.inference_mode()
    def test_qkv_fusion_linear_longformer_model(self, dtype, freeze_opt=True):
        """Test QKV fusion with Longformer-style pattern (permutations and as_strided)"""
        torch.manual_seed(42)

        # Create model with Longformer-style QKV pattern
        model = Custom_Model_QKV_Linear_Longformer(
            dtype=DataTypes.get_torch_type(dtype), hidden_size=768, num_heads=12
        )
        # Create input tensor with shape (batch_size, seq_len, hidden_size)
        input_tensor = torch.randn(1, 512, 768, dtype=DataTypes.get_torch_type(dtype))

        # Inductor backend with freezing
        reset_dynamo()
        inductor_model = torch.compile(
            copy.deepcopy(model), backend="inductor", options={"freezing": True}
        )
        inductor_output = test_with_freeze_opt(
            inductor_model, (input_tensor,), freeze_opt=True
        )

        # Zentorch backend with freezing
        reset_dynamo()
        counters.clear()
        self.assertEqual(counters["zentorch"]["qkv_fusion_linear"], 0)
        self.assertEqual(counters["zentorch"]["qkv_fusion_linear_contiguous"], 0)
        zentorch_model = torch.compile(copy.deepcopy(model), backend="zentorch")
        zentorch_output = test_with_freeze_opt(
            zentorch_model, (input_tensor,), freeze_opt=True
        )
        self.assertEqual(counters["zentorch"]["qkv_fusion_linear"], 1)
        self.assertEqual(counters["zentorch"]["qkv_fusion_linear_contiguous"], 1)

        # Compare both attn and v outputs
        self.assertTrue(
            torch.allclose(
                inductor_output[0], zentorch_output[0], rtol=1e-3, atol=1e-3
            ),
            f"Attention tensors don't match. Max diff: {(inductor_output[0] - zentorch_output[0]).abs().max().item()}",
        )
        self.assertTrue(
            torch.allclose(
                inductor_output[1], zentorch_output[1], rtol=1e-3, atol=1e-3
            ),
            f"V tensors don't match. Max diff: {(inductor_output[1] - zentorch_output[1]).abs().max().item()}",
        )

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes,
        freeze_list=[True],
    )
    @torch.inference_mode()
    def test_qkv_fusion_linear_skip_prepacked_weights(self, dtype, freeze_opt=True):
        """QKV fusion must be skipped when weights are pre-packed."""
        model = Custom_Model_QKV_Linear_ZenTorch(
            dtype=DataTypes.get_torch_type(dtype),
            input_dim=self.data.k,
            hidden_dim=self.data.n,
            is_weight_prepacked=True,
        )
        input_tensor = torch.randn(
            self.data.b, self.data.m, self.data.k,
            dtype=DataTypes.get_torch_type(dtype),
        )

        reset_dynamo()
        counters.clear()
        compiled_graph = torch.compile(model, backend="zentorch")
        with self.assertLogs("zentorch", level="INFO") as cm:
            test_with_freeze_opt(
                compiled_graph, (input_tensor,), freeze_opt=True
            )
        self.assertEqual(counters["zentorch"]["qkv_fusion_linear"], 0)
        self.assertTrue(
            any("prepacked weights" in msg for msg in cm.output),
            f"Expected log about prepacked weights, got: {cm.output}",
        )

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes,
        freeze_list=[True],
    )
    @torch.inference_mode()
    def test_qkv_fusion_linear_skip_inconsistent_post_ops(self, dtype, freeze_opt=True):
        """QKV fusion must be skipped when linears have different post-ops."""
        model = Custom_Model_QKV_Linear_ZenTorch(
            dtype=DataTypes.get_torch_type(dtype),
            input_dim=self.data.k,
            hidden_dim=self.data.n,
            post_ops=("relu", "gelu_erf", "relu"),
        )
        input_tensor = torch.randn(
            self.data.b, self.data.m, self.data.k,
            dtype=DataTypes.get_torch_type(dtype),
        )

        native_output = model(input_tensor)
        reset_dynamo()
        counters.clear()
        compiled_graph = torch.compile(model, backend="zentorch")
        with self.assertLogs("zentorch", level="INFO") as cm:
            compiled_output = test_with_freeze_opt(
                compiled_graph, (input_tensor,), freeze_opt=True
            )
        self.assertEqual(counters["zentorch"]["qkv_fusion_linear"], 0)
        self.assertTrue(
            any("inconsistent post-ops" in msg for msg in cm.output),
            f"Expected log about inconsistent post-ops, got: {cm.output}",
        )
        self.assertEqual(native_output, compiled_output)

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes,
        freeze_list=[True],
    )
    @torch.inference_mode()
    def test_qkv_fusion_linear_same_post_op_propagation(self, dtype, freeze_opt=True):
        """When all 3 linears share the same post-op, fusion should happen
        and the post-op must be propagated to the fused linear correctly."""
        model = Custom_Model_QKV_Linear_ZenTorch(
            dtype=DataTypes.get_torch_type(dtype),
            input_dim=self.data.k,
            hidden_dim=self.data.n,
            post_ops=("relu", "relu", "relu"),
        )
        input_tensor = torch.randn(
            self.data.b, self.data.m, self.data.k,
            dtype=DataTypes.get_torch_type(dtype),
        )

        native_output = model(input_tensor)
        reset_dynamo()
        counters.clear()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(
            compiled_graph, (input_tensor,), freeze_opt=True
        )
        self.assertEqual(counters["zentorch"]["qkv_fusion_linear"], 1)
        self.assertEqual(native_output, compiled_output)


if __name__ == "__main__":
    run_tests()
