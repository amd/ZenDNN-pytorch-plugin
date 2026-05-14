# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: E402
    Zentorch_TestCase,
    has_zentorch,
    zentorch,
    run_tests,
)

# Todo: Below way of defining the configs will be cleaned up.
# (num_experts, M, K, N_or_D, dtype)
PLAIN_GEMM_CONFIGS = [
    (8, 4, 64, 32, torch.bfloat16),
    (4, 8, 64, 32, torch.float32),
]

# (num_experts, K, N, topk, num_tokens, dtype)
MOE_REDUCE_CONFIGS = [
    (4, 64, 32, 2, 8, torch.float32),
    (4, 16, 8, 2, 8, torch.bfloat16),
]

# (num_experts, M, K, D, activation, dtype)
GATED_ACTIVATION_CONFIGS = [
    (4, 8, 64, 32, "gelu", torch.float32),
    (4, 8, 64, 32, "swigluoai", torch.float32),
    (4, 8, 32, 32, "silu", torch.bfloat16),
]

# (num_experts, M, K, D, K_out, activation, dtype)
FUSED_W2_CONFIGS = [
    (4, 8, 64, 32, 64, "silu", torch.float32),
]

# (num_experts, K, D, K_out, topk, num_tokens, activation, dtype)
FULL_MOE_FFN_CONFIGS = [
    (4, 32, 16, 32, 2, 8, "silu", torch.bfloat16),
    (4, 32, 16, 32, 2, 8, "swigluoai", torch.bfloat16),
]

# Tolerance per dtype
TOLERANCES = {
    torch.float32: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 1e-2, "rtol": 1e-2},
    "fused_bf16": {"atol": 5e-1, "rtol": 5e-1},
}


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GroupMatmul(Zentorch_TestCase):
    """Parameterized tests for zentorch_group_matmul.out."""

    # ------------------------------------------------------------------
    # Reference helpers
    # ------------------------------------------------------------------

    def _reference_weighted_reduce(self, expert_outputs, topk_weights,
                                   topk_indices, num_tokens, topk):
        """Reference: output[t,d] = sum_k w[t,k] * expert_dst[eid][row_j][d]."""
        hidden_dim = expert_outputs[0].shape[1]
        num_experts = len(expert_outputs)
        expert_count = [0] * num_experts
        output = torch.zeros(num_tokens, hidden_dim, dtype=expert_outputs[0].dtype)
        for t in range(num_tokens):
            for k in range(topk):
                expert_id = topk_indices[t, k].item()
                row_j = expert_count[expert_id]
                expert_count[expert_id] += 1
                output[t] += topk_weights[t, k].item() * expert_outputs[expert_id][row_j]
        return output

    def _scatter_and_build_row_ptrs(self, hidden_states, topk_indices,
                                    num_experts, K, N):
        """Scatter tokens to experts, pre-allocate outputs, build row_ptrs."""
        num_tokens = hidden_states.shape[0]
        topk = topk_indices.shape[1]
        expert_token_counts = [(topk_indices == e).sum().item() for e in range(num_experts)]
        inputs = [torch.zeros(int(expert_token_counts[e]), K, dtype=hidden_states.dtype)
                  for e in range(num_experts)]
        gemm_outputs = [torch.empty(int(expert_token_counts[e]), N, dtype=hidden_states.dtype)
                        for e in range(num_experts)]
        row_ptrs = torch.zeros(num_tokens * topk, dtype=torch.int64)
        expert_cursor = [0] * num_experts
        for t in range(num_tokens):
            for k in range(topk):
                expert_id = topk_indices[t, k].item()
                row_j = expert_cursor[expert_id]
                inputs[expert_id][row_j] = hidden_states[t]
                row_ptrs[t * topk + k] = (
                    gemm_outputs[expert_id].data_ptr()
                    + row_j * gemm_outputs[expert_id].stride(0)
                    * gemm_outputs[expert_id].element_size())
                expert_cursor[expert_id] += 1
        return inputs, gemm_outputs, row_ptrs

    def _reference_expert_outputs(self, inputs, weights, bias, activation="none",
                                  w2_weights=None, w2_bias=None, compute_in_fp32=False):
        """Per-expert reference: linear → optional activation → optional w2."""
        expert_outputs = []
        for i in range(len(inputs)):
            expert_input = inputs[i].float() if compute_in_fp32 else inputs[i]
            expert_weight = weights[i].float() if compute_in_fp32 else weights[i]
            expert_bias = bias[i].float() if (bias[i] is not None and compute_in_fp32) else bias[i]
            result = torch.nn.functional.linear(expert_input, expert_weight, expert_bias)
            if activation != "none":
                result = self._apply_gated_activation(result, activation)
            if w2_weights is not None:
                down_weight = w2_weights[i].float() if compute_in_fp32 else w2_weights[i]
                down_bias = w2_bias[i].float() if (w2_bias[i] is not None and compute_in_fp32) else w2_bias[i]
                result = torch.nn.functional.linear(result, down_weight, down_bias)
            expert_outputs.append(result)
        return expert_outputs

    def _apply_gated_activation(self, tensor, activation):
        """Gated activation: split → act(gate) * value. Input [M, 2*D] → [M, D]."""
        self.assertEqual(tensor.shape[1] % 2, 0,
                         f"Gated activation requires even last dim, got {tensor.shape[1]}")
        half_dim = tensor.shape[1] // 2
        gate = tensor[:, :half_dim]
        value = tensor[:, half_dim:]
        if activation == "silu":
            return torch.nn.functional.silu(gate) * value
        elif activation == "gelu":
            return torch.nn.functional.gelu(gate) * value
        elif activation == "swigluoai":
            alpha, limit = 1.702, 7.0
            gate, up = tensor[..., ::2], tensor[..., 1::2]
            gate = gate.clamp(min=None, max=limit)
            up = up.clamp(min=-limit, max=limit)
            return (up + 1) * (gate * torch.sigmoid(gate * alpha))
        raise ValueError(f"Unsupported activation: {activation!r}")

# Todo: Update the testcases in a parameterized structure.
# JIRA ID: ZENAI-3656
    @torch.inference_mode()
    def test_plain_gemm(self):
        """Plain parallel GEMM across dtypes and dimensions."""
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")

        for num_experts, M, K, N, dtype in PLAIN_GEMM_CONFIGS:
            bias = [None] * num_experts
            inputs = [torch.randn(M, K, dtype=dtype) for i in range(num_experts)]
            weights = [torch.randn(N, K, dtype=dtype) for i in range(num_experts)]

            ref = self._reference_expert_outputs(inputs, weights, bias)

            gemm_outputs = [torch.empty(M, N, dtype=dtype) for i in range(num_experts)]
            torch.ops.zentorch.zentorch_group_matmul.out(
                gemm_outputs, inputs, weights, bias, "none", [], [])

            tol = TOLERANCES[dtype]
            for i in range(num_experts):
                self.assertEqual(gemm_outputs[i], ref[i], **tol)

    @torch.inference_mode()
    def test_moe_weighted_reduce(self):
        """GEMM + MoE weighted-reduce across dtypes and dimensions."""
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")

        for num_experts, K, N, topk, num_tokens, dtype in MOE_REDUCE_CONFIGS:
            bias = [None] * num_experts
            topk_indices = torch.randint(0, num_experts, (num_tokens, topk))
            topk_weights = torch.rand(num_tokens, topk)
            hidden_states = torch.randn(num_tokens, K, dtype=dtype)

            moe_inputs, moe_gemm_outputs, row_ptrs = self._scatter_and_build_row_ptrs(
                hidden_states, topk_indices, num_experts, K, N)
            weights = [torch.randn(N, K, dtype=dtype) for i in range(num_experts)]
            moe_output = torch.empty(num_tokens, N, dtype=dtype)

            is_bf16 = (dtype == torch.bfloat16)
            ref_experts = self._reference_expert_outputs(
                moe_inputs, weights, bias, compute_in_fp32=is_bf16)
            ref_moe = self._reference_weighted_reduce(
                ref_experts, topk_weights, topk_indices, num_tokens, topk)

            torch.ops.zentorch.zentorch_group_matmul.out(
                moe_gemm_outputs, moe_inputs, weights, bias, "none", [], [],
                moe_output, topk_weights, row_ptrs)

            tol = TOLERANCES[dtype]
            actual = moe_output.float() if is_bf16 else moe_output
            self.assertEqual(actual, ref_moe, **tol)

    @torch.inference_mode()
    def test_gated_activations(self):
        """GEMM + gated activation across activation types and dtypes."""
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")

        for num_experts, M, K, D, activation, dtype in GATED_ACTIVATION_CONFIGS:
            N = 2 * D
            bias = [None] * num_experts
            inputs = [torch.randn(M, K, dtype=dtype) for i in range(num_experts)]
            weights = [torch.randn(N, K, dtype=dtype) for i in range(num_experts)]

            ref = self._reference_expert_outputs(inputs, weights, bias, activation=activation)

            gemm_outputs = [torch.empty(M, N, dtype=dtype) for i in range(num_experts)]
            torch.ops.zentorch.zentorch_group_matmul.out(
                gemm_outputs, inputs, weights, bias, activation, [], [])

            tol = TOLERANCES[dtype]
            for i in range(num_experts):
                self.assertEqual(gemm_outputs[i][:, :D], ref[i], **tol)

    @torch.inference_mode()
    def test_unsupported_activation(self):
        """Unsupported activation string raises RuntimeError."""
        num_experts, M, K, N = 4, 8, 64, 32
        inputs = [torch.randn(M, K) for i in range(num_experts)]
        weights = [torch.randn(N, K) for i in range(num_experts)]
        bias = [None] * num_experts
        gemm_outputs = [torch.empty(M, N) for i in range(num_experts)]

        for bad_activation in ["relu", "tanh"]:
            with self.assertRaisesRegex(RuntimeError, "unsupported activation"):
                torch.ops.zentorch.zentorch_group_matmul.out(
                    gemm_outputs, inputs, weights, bias, bad_activation, [], [])

    @torch.inference_mode()
    def test_fused_w2(self):
        """Fused w13 → activation → w2 (with bias) across configs."""
        for num_experts, M, K, D, K_out, activation, dtype in FUSED_W2_CONFIGS:
            N = 2 * D
            w13_bias = [None] * num_experts
            inputs = [torch.randn(M, K, dtype=dtype) for i in range(num_experts)]
            w13_weights = [torch.randn(N, K, dtype=dtype) for i in range(num_experts)]
            w2_weights = [torch.randn(K_out, D, dtype=dtype) for i in range(num_experts)]
            w2_bias = [torch.randn(K_out, dtype=dtype) for i in range(num_experts)]

            gemm_outputs = [torch.empty(M, N, dtype=dtype) for i in range(num_experts)]

            # ZenDNN manages w2 output internally
            torch.ops.zentorch.zentorch_group_matmul.out(
                gemm_outputs, inputs, w13_weights, w13_bias, activation,
                w2_weights, w2_bias)

    @torch.inference_mode()
    def test_empty_gemm_outputs_fused_w2(self):
        """Fused w2 with gemm_outputs=[] — backend allocates dst internally."""
        for num_experts, M, K, D, K_out, activation, dtype in FUSED_W2_CONFIGS:
            N = 2 * D
            w13_bias = [None] * num_experts
            inputs = [torch.randn(M, K, dtype=dtype) for i in range(num_experts)]
            w13_weights = [torch.randn(N, K, dtype=dtype) for i in range(num_experts)]
            w2_weights = [torch.randn(K_out, D, dtype=dtype) for i in range(num_experts)]
            w2_bias = [torch.randn(K_out, dtype=dtype) for i in range(num_experts)]

            # ZenDNN manages w2 output internally
            torch.ops.zentorch.zentorch_group_matmul.out(
                [], inputs, w13_weights, w13_bias, activation,
                w2_weights, w2_bias)

    @torch.inference_mode()
    def test_fused_moe_pipeline(self):
        """Full pipeline: w13 → activation → w2 (with bias) → MoE reduce, bf16.

        Two output paths are verified per config:
          - Low-level: zentorch_group_matmul.out with inline MoE weighted-reduce
          - High-level: zentorch_fused_moe (token grouping + W13 + act + W2 +
            weighted reduce in a single op call)

        ZenDNN reuses the input buffers for w2 output, so row_ptrs must
        point into the input tensors (which will hold the down projection
        results after the kernel completes).
        """
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")

        for (num_experts, K, D, K_out, topk, num_tokens,
             activation, dtype) in FULL_MOE_FFN_CONFIGS:
            N = 2 * D
            w13_bias = [None] * num_experts
            w13_weights = [torch.randn(N, K, dtype=dtype) for i in range(num_experts)]
            w2_weights = [torch.randn(K_out, D, dtype=dtype) for i in range(num_experts)]
            w2_bias = [torch.randn(K_out, dtype=dtype) for i in range(num_experts)]

            topk_indices = torch.randint(0, num_experts, (num_tokens, topk))
            topk_weights_t = torch.rand(num_tokens, topk)
            hidden_states = torch.randn(num_tokens, K, dtype=dtype)

            # Scatter tokens to per-expert batches. ZenDNN reuses input
            # buffers for w2 output, so row_ptrs must point into inputs.
            inputs, unused_outputs, unused_ptrs = self._scatter_and_build_row_ptrs(
                hidden_states, topk_indices, num_experts, K, K_out)

            # Todo: Will try to reuse the existing function to build row_ptrs.
            # Build row_ptrs pointing into input buffers (MoE reduce target)
            row_ptrs_into_inputs = torch.zeros(num_tokens * topk, dtype=torch.int64)
            per_expert_row = [0] * num_experts
            for token_idx in range(num_tokens):
                for topk_idx in range(topk):
                    expert_id = topk_indices[token_idx, topk_idx].item()
                    row_in_expert = per_expert_row[expert_id]
                    row_ptrs_into_inputs[token_idx * topk + topk_idx] = (
                        inputs[expert_id].data_ptr()
                        + row_in_expert * inputs[expert_id].stride(0)
                        * inputs[expert_id].element_size())
                    per_expert_row[expert_id] += 1

            ref_down = self._reference_expert_outputs(
                inputs, w13_weights, w13_bias, activation=activation,
                w2_weights=w2_weights, w2_bias=w2_bias,
                compute_in_fp32=(dtype == torch.bfloat16))
            ref_moe = self._reference_weighted_reduce(
                ref_down, topk_weights_t, topk_indices, num_tokens, topk)

            # --- Output 1: low-level zentorch_group_matmul.out ---
            moe_reduce_output = torch.empty(num_tokens, K_out, dtype=dtype)
            gate_up_outputs = [torch.empty(inputs[i].size(0), N, dtype=dtype)
                               for i in range(num_experts)]

            torch.ops.zentorch.zentorch_group_matmul.out(
                gate_up_outputs, inputs, w13_weights, w13_bias, activation,
                w2_weights, w2_bias,
                moe_reduce_output, topk_weights_t, row_ptrs_into_inputs)

            self.assertEqual(moe_reduce_output.float(), ref_moe, **TOLERANCES["fused_bf16"])

            # --- Output 2: high-level zentorch_fused_moe op ---
            # Wraps Phase 1 (token grouping) + W13 GEMM + gated activation +
            # W2 GEMM + weighted reduce in a single call. Inputs are the
            # un-scattered hidden_states + 3-D stacked weights; output must
            # match the same per-expert reference reduction (ref_moe).
            w13_3d = torch.stack(w13_weights, dim=0)          # [E, 2*D, K]
            w2_3d = torch.stack(w2_weights, dim=0)            # [E, K_out, D]
            w2_bias_3d = torch.stack(w2_bias, dim=0)          # [E, K_out]

            fused_moe_output = torch.zeros(num_tokens, K_out, dtype=dtype)
            torch.ops.zentorch.zentorch_fused_moe(
                fused_moe_output,
                hidden_states,
                w13_3d, w2_3d,
                None,                                          # no w13_bias
                w2_bias_3d,
                topk_weights_t, topk_indices.to(torch.int32),
                False,                                         # skip_weighted
                activation,
            )

            self.assertEqual(fused_moe_output.shape, (num_tokens, K_out))
            self.assertEqual(fused_moe_output.float(), ref_moe, **TOLERANCES["fused_bf16"])


if __name__ == "__main__":
    run_tests()
