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


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GroupMatmul(Zentorch_TestCase):
    """Test zentorch_group_matmul.out parallel mode with optional MoE weighted-reduce."""

    def _reference_weighted_reduce(self, expert_outputs, topk_weights,
                                   topk_indices, num_tokens, topk):
        """Reference: output[t,d] = sum_k w[t,k] * expert_dst[eid][row_j][d]."""
        hidden_dim = expert_outputs[0].shape[1]
        num_experts = len(expert_outputs)
        # Track how many tokens each expert has processed so far
        expert_count = [0] * num_experts
        output = torch.zeros(num_tokens, hidden_dim, dtype=expert_outputs[0].dtype)
        for t in range(num_tokens):
            for k in range(topk):
                expert_id = topk_indices[t, k].item()
                row_j = expert_count[expert_id]
                expert_count[expert_id] += 1
                # Accumulate weighted expert result for this token
                output[t] += topk_weights[t, k].item() * expert_outputs[expert_id][row_j]
        return output

    def _scatter_and_build_row_ptrs(self, hidden_states, topk_indices,
                                    num_experts, K, N):
        """Scatter tokens to experts, pre-allocate gemm_outputs, and build row_ptrs.

        Returns (inputs, gemm_outputs, row_ptrs).
        """
        num_tokens = hidden_states.shape[0]
        topk = topk_indices.shape[1]

        # Count how many tokens are routed to each expert
        expert_token_counts = [(topk_indices == e).sum().item() for e in range(num_experts)]

        # Per-expert input batches [M_i, K] and pre-allocated output buffers [M_i, N]
        inputs = [torch.zeros(int(expert_token_counts[e]), K, dtype=hidden_states.dtype)
                  for e in range(num_experts)]
        gemm_outputs = [torch.empty(int(expert_token_counts[e]), N, dtype=hidden_states.dtype)
                        for e in range(num_experts)]

        # Build row_ptrs (int64 pointer addresses) alongside the scatter
        row_ptrs = torch.zeros(num_tokens * topk, dtype=torch.int64)
        expert_cursor = [0] * num_experts
        for t in range(num_tokens):
            for k in range(topk):
                expert_id = topk_indices[t, k].item()
                row_j = expert_cursor[expert_id]
                # Scatter: copy token into the expert's input batch
                inputs[expert_id][row_j] = hidden_states[t]
                # Store raw pointer to the corresponding row in gemm_outputs
                row_ptrs[t * topk + k] = (
                    gemm_outputs[expert_id].data_ptr()
                    + row_j * gemm_outputs[expert_id].stride(0)
                    * gemm_outputs[expert_id].element_size())
                expert_cursor[expert_id] += 1

        return inputs, gemm_outputs, row_ptrs

    # Todo: Include testcases for
    # 1. BF16 Grouped GEMM + Weighted-reduce post-op
    # 2. FP32 Grouped GEMM
    @torch.inference_mode()
    def test_parallel_bf16(self):
        """Parallel group matmul, bfloat16."""
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")

        # 8 experts, each processing 4 tokens with K=64 input features and N=32 output features
        num_experts, M, K, N = 8, 4, 64, 32
        inputs = [torch.randn(M, K, dtype=torch.bfloat16) for i in range(num_experts)]
        weights = [torch.randn(N, K, dtype=torch.bfloat16) for i in range(num_experts)]
        bias = [None] * num_experts

        # Compute reference using torch.nn.functional.linear per expert
        ref_expert_outputs = []
        for i in range(num_experts):
            ref_expert_outputs.append(torch.nn.functional.linear(inputs[i], weights[i], None))

        # Pre-allocate output buffers and run the operator
        gemm_outputs = [torch.empty(M, N, dtype=torch.bfloat16) for i in range(num_experts)]
        torch.ops.zentorch.zentorch_group_matmul.out(gemm_outputs, inputs, weights, bias)

        # Verify each expert's output matches the reference
        self.assertEqual(len(gemm_outputs), num_experts)
        for i in range(num_experts):
            torch.testing.assert_close(gemm_outputs[i], ref_expert_outputs[i], atol=1e-2, rtol=1e-2)

    @torch.inference_mode()
    def test_moe_weighted_reduce_fp32(self):
        """Parallel group matmul + MoE weighted-reduce post-op, fp32."""
        num_experts, K, N = 4, 64, 32
        topk = 2
        num_tokens = 8

        # Simulate router: random expert assignments and routing weights
        topk_indices = torch.randint(0, num_experts, (num_tokens, topk))
        topk_weights = torch.rand(num_tokens, topk)
        hidden_states = torch.randn(num_tokens, K)

        # Scatter tokens to experts, pre-allocate gemm_outputs, build row_ptrs
        inputs, gemm_outputs, row_ptrs = self._scatter_and_build_row_ptrs(
            hidden_states, topk_indices, num_experts, K, N)

        weights = [torch.randn(N, K) for i in range(num_experts)]
        bias = [None] * num_experts
        # Pre-allocate the MoE reduced output [num_tokens, N]
        moe_output = torch.empty(num_tokens, N)

        # Compute reference: per-expert linear + weighted-reduce
        ref_expert_outputs = []
        for i in range(num_experts):
            ref_expert_outputs.append(torch.nn.functional.linear(inputs[i], weights[i], None))
        ref_moe_output = self._reference_weighted_reduce(
            ref_expert_outputs, topk_weights, topk_indices, num_tokens, topk)

        # Run the operator with MoE post-op
        torch.ops.zentorch.zentorch_group_matmul.out(
            gemm_outputs, inputs, weights, bias,
            moe_output=moe_output, topk_weights=topk_weights, row_ptrs=row_ptrs)

        # Verify the fused MoE output matches the reference
        self.assertEqual(moe_output.shape, (num_tokens, N))
        torch.testing.assert_close(moe_output, ref_moe_output, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    run_tests()
