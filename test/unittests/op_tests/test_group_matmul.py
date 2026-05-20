# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
from pathlib import Path
from parameterized import parameterized
from itertools import product

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: E402
    Zentorch_TestCase,
    has_zentorch,
    zentorch,
    run_tests,
    supported_dtypes,
)

GROUP_MATMUL_CONFIGS = {
    "num_experts": [4],
    "M": [8],
    "K": [64],
    "N": [32],
    "D": [16, 32],
    "K_out": [32, 64],
    "topk": [2],
    "num_tokens": [8],
}

SUPPORTED_ACTIVATIONS = ["silu", "gelu", "swigluoai"]

# Tolerance per dtype
TOLERANCES = {
    torch.float32: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 3e-2, "rtol": 3e-2},
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
        """Gated activation: split → act(gate) * value. Input [M, 2*D] → [M, D].

        Matches cpp/GroupMatmul.cpp::map_activation_to_gated_act:
          "none" / ""       → none
          "silu"    → silu_and_mul
          "gelu"    → gelu_and_mul
          "swigluoai"  → swiglu_oai_mul
        """
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

    @parameterized.expand(product(
        supported_dtypes,
        GROUP_MATMUL_CONFIGS["num_experts"],
        GROUP_MATMUL_CONFIGS["M"],
        GROUP_MATMUL_CONFIGS["K"],
        GROUP_MATMUL_CONFIGS["N"],
    ))
    @torch.inference_mode()
    def test_plain_gemm(self, dtype_str, num_experts, M, K, N):
        """Plain parallel GEMM."""
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")

        torch.manual_seed(42)
        dtype = self.data.get_torch_type(dtype_str)
        w13_bias = [None] * num_experts
        inputs = [torch.randn(M, K, dtype=dtype) for i in range(num_experts)]
        w13 = [torch.randn(N, K, dtype=dtype) for i in range(num_experts)]

        ref = self._reference_expert_outputs(inputs, w13, w13_bias)

        gemm_outputs = [torch.empty(M, N, dtype=dtype) for i in range(num_experts)]
        torch.ops.zentorch.zentorch_group_matmul.out(
            gemm_outputs, inputs, w13, [], None, None, None,
            "none", w13_bias, [], [], [])

        tol = TOLERANCES[dtype]
        for i in range(num_experts):
            self.assertEqual(gemm_outputs[i], ref[i], **tol)

    @parameterized.expand(product(
        supported_dtypes,
        GROUP_MATMUL_CONFIGS["num_experts"],
        GROUP_MATMUL_CONFIGS["K"],
        GROUP_MATMUL_CONFIGS["N"],
        GROUP_MATMUL_CONFIGS["topk"],
        GROUP_MATMUL_CONFIGS["num_tokens"],
    ))
    @torch.inference_mode()
    def test_moe_weighted_reduce(self, dtype_str, num_experts, K, N, topk, num_tokens):
        """GEMM + MoE weighted-reduce."""
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")

        torch.manual_seed(42)
        dtype = self.data.get_torch_type(dtype_str)
        w13_bias = [None] * num_experts
        topk_indices = torch.randint(0, num_experts, (num_tokens, topk))
        topk_weights = torch.rand(num_tokens, topk)
        hidden_states = torch.randn(num_tokens, K, dtype=dtype)

        inputs, moe_gemm_outputs, row_ptrs = self._scatter_and_build_row_ptrs(
            hidden_states, topk_indices, num_experts, K, N)
        w13 = [torch.randn(N, K, dtype=dtype) for i in range(num_experts)]
        moe_output = torch.empty(num_tokens, N, dtype=dtype)

        is_bf16 = (dtype == torch.bfloat16)
        ref_experts = self._reference_expert_outputs(
            inputs, w13, w13_bias, compute_in_fp32=is_bf16)
        ref_moe = self._reference_weighted_reduce(
            ref_experts, topk_weights, topk_indices, num_tokens, topk)

        torch.ops.zentorch.zentorch_group_matmul.out(
            moe_gemm_outputs, inputs, w13, [],
            moe_output, topk_weights, row_ptrs,
            "none", w13_bias, [], [], [])

        tol = TOLERANCES[dtype]
        actual = moe_output.float() if is_bf16 else moe_output
        self.assertEqual(actual, ref_moe, **tol)

    @parameterized.expand(product(
        supported_dtypes,
        GROUP_MATMUL_CONFIGS["num_experts"],
        GROUP_MATMUL_CONFIGS["M"],
        GROUP_MATMUL_CONFIGS["K"],
        GROUP_MATMUL_CONFIGS["D"],
    ))
    @torch.inference_mode()
    def test_gated_activations(self, dtype_str, num_experts, M, K, D):
        """GEMM + gated activation across activation types."""
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")

        torch.manual_seed(42)
        dtype = self.data.get_torch_type(dtype_str)
        N = 2 * D
        for activation in SUPPORTED_ACTIVATIONS:
            w13_bias = [None] * num_experts
            inputs = [torch.randn(M, K, dtype=dtype) for i in range(num_experts)]
            w13 = [torch.randn(N, K, dtype=dtype) for i in range(num_experts)]

            ref = self._reference_expert_outputs(inputs, w13, w13_bias, activation=activation)

            gemm_outputs = [torch.empty(M, N, dtype=dtype) for i in range(num_experts)]
            torch.ops.zentorch.zentorch_group_matmul.out(
                gemm_outputs, inputs, w13, [], None, None, None,
                activation, w13_bias, [], [], [])

            tol = TOLERANCES[dtype]
            for i in range(num_experts):
                self.assertEqual(gemm_outputs[i][:, :D], ref[i], **tol)

    @parameterized.expand(supported_dtypes)
    @torch.inference_mode()
    def test_unsupported_activation(self, dtype_str):
        """Unsupported activation string raises RuntimeError."""
        dtype = self.data.get_torch_type(dtype_str)
        num_experts, M, K, N = 4, 8, 64, 32
        inputs = [torch.randn(M, K, dtype=dtype) for i in range(num_experts)]
        w13 = [torch.randn(N, K, dtype=dtype) for i in range(num_experts)]
        w13_bias = [None] * num_experts
        gemm_outputs = [torch.empty(M, N, dtype=dtype) for i in range(num_experts)]

        for bad_activation in ["relu", "tanh"]:
            with self.assertRaisesRegex(RuntimeError, "unsupported activation"):
                torch.ops.zentorch.zentorch_group_matmul.out(
                    gemm_outputs, inputs, w13, [], None, None, None,
                    bad_activation, w13_bias, [], [], [])

    @parameterized.expand(product(
        supported_dtypes,
        GROUP_MATMUL_CONFIGS["num_experts"],
        GROUP_MATMUL_CONFIGS["M"],
        GROUP_MATMUL_CONFIGS["K"],
        GROUP_MATMUL_CONFIGS["N"],
    ))
    @torch.inference_mode()
    def test_int8_w13(self, dtype_str, num_experts, M, K, N):
        """Dynamic int8: inputs x w13 int8 weights with per-channel scales."""
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")
        if not zentorch._C.is_bf16_supported():
            self.skipTest("BF16 not supported")

        torch.manual_seed(42)
        dtype = self.data.get_torch_type(dtype_str)
        inputs = [torch.randn(M, K, dtype=dtype) for i in range(num_experts)]

        weights_fp32 = [torch.randn(N, K) for i in range(num_experts)]
        w13_int8 = []
        w13_scales = []
        for i in range(num_experts):
            scales = weights_fp32[i].abs().amax(dim=1).clamp(min=1e-12) / 127.0
            zero_points = torch.zeros(N, dtype=torch.long)
            weight_q = torch.quantize_per_channel(
                weights_fp32[i], scales, zero_points, axis=0, dtype=torch.qint8)
            w13_int8.append(weight_q.int_repr())
            w13_scales.append(weight_q.q_per_channel_scales().to(torch.float32))
        w13_bias = [None] * num_experts

        ref = []
        for i in range(num_experts):
            w_deq = w13_int8[i].float() * w13_scales[i].unsqueeze(1)
            ref.append(torch.nn.functional.linear(inputs[i].float(), w_deq, None))

        gemm_outputs = [torch.empty(M, N, dtype=dtype) for i in range(num_experts)]
        torch.ops.zentorch.zentorch_group_matmul.out(
            gemm_outputs, inputs, w13_int8, [], None, None, None,
            "none", w13_bias, [], w13_scales, [])

        for i in range(num_experts):
            self.assertEqual(gemm_outputs[i].float(), ref[i], atol=2e-1, rtol=2e-1)

    @parameterized.expand(product(
        ["float32"],
        GROUP_MATMUL_CONFIGS["num_experts"],
        GROUP_MATMUL_CONFIGS["K"],
        GROUP_MATMUL_CONFIGS["topk"],
        GROUP_MATMUL_CONFIGS["num_tokens"],
    ))
    @torch.inference_mode()
    def test_int8_w13_and_w2(self, dtype_str, num_experts, K, topk, num_tokens):
        """Dynamic int8 w13 + int8 w2 with per-channel scales via zentorch_fused_moe.

        Exercises the full MoE chain (W13 -> gated activation -> W2 -> weighted
        reduce) for int8 weights. Goes through zentorch_fused_moe so that, on
        the int8 path, the internal split-call workaround for the ZenDNN
        fused-chain bug (every-other-row scrambling under int8 + gated
        activation + M>1) is exercised.

        Constraint: K == K_out (W2 output dim).
        """
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")
        if not zentorch._C.is_bf16_supported():
            self.skipTest("BF16 not supported")

        torch.manual_seed(42)
        dtype = self.data.get_torch_type(dtype_str)
        activation = "silu"
        gated = activation in ("silu", "gelu", "swigluoai")
        D = K // 2 if gated else K
        N = 2 * D if gated else D
        K_out = K

        def quantize_weights(fp32_list):
            int8_list, scale_list = [], []
            for w in fp32_list:
                s = w.abs().amax(dim=1).clamp(min=1e-12) / 127.0
                zp = torch.zeros(w.size(0), dtype=torch.long)
                wq = torch.quantize_per_channel(w, s, zp, axis=0, dtype=torch.qint8)
                int8_list.append(wq.int_repr())
                scale_list.append(wq.q_per_channel_scales().to(torch.float32))
            return int8_list, scale_list

        def dynamic_quant_matmul(src, w_int8, w_scales, bias=None):
            src_fp = src.float()
            src_scale = src_fp.abs().amax(dim=1).clamp(min=1e-12) / 127.0
            src_q = (src_fp / src_scale.unsqueeze(1)).round().clamp(-128, 127)
            acc = src_q @ w_int8.float().T
            result = acc * (src_scale.unsqueeze(1) * w_scales.unsqueeze(0))
            if bias is not None:
                result = result + bias.float()
            return result

        def apply_gated_activation(x, act):
            if act == "none":
                return x
            gate, up = x.chunk(2, dim=-1)
            if act == "silu":
                return torch.nn.functional.silu(gate) * up
            if act == "gelu":
                return torch.nn.functional.gelu(gate) * up
            raise ValueError(f"unsupported activation: {act}")

        w13_fp32 = [torch.randn(N, K) for i in range(num_experts)]
        w13_int8, w13_scales = quantize_weights(w13_fp32)

        w2_fp32 = [torch.randn(K_out, D) for i in range(num_experts)]
        w2_int8, w2_scales = quantize_weights(w2_fp32)
        w2_bias = [torch.randn(K_out, dtype=dtype) for i in range(num_experts)]

        hidden_states = torch.randn(num_tokens, K, dtype=dtype)
        topk_indices = torch.randint(0, num_experts, (num_tokens, topk))
        topk_weights_t = torch.rand(num_tokens, topk)

        inputs, unused_outputs, unused_ptrs = self._scatter_and_build_row_ptrs(
            hidden_states, topk_indices, num_experts, K, K_out)

        ref_experts = []
        for i in range(num_experts):
            r = dynamic_quant_matmul(inputs[i], w13_int8[i], w13_scales[i])
            r = apply_gated_activation(r, activation)
            r = dynamic_quant_matmul(r, w2_int8[i], w2_scales[i],
                                     bias=w2_bias[i])
            ref_experts.append(r)
        ref_moe = self._reference_weighted_reduce(
            ref_experts, topk_weights_t, topk_indices, num_tokens, topk)

        w13_3d = torch.stack(w13_int8, dim=0)
        w2_3d = torch.stack(w2_int8, dim=0)
        w13_scales_3d = torch.stack(w13_scales, dim=0)
        w2_scales_3d = torch.stack(w2_scales, dim=0)
        w2_bias_3d = torch.stack(w2_bias, dim=0)

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
            w13_scales_3d, w2_scales_3d,
        )

        self.assertEqual(fused_moe_output.shape, (num_tokens, K_out))
        self.assertEqual(fused_moe_output.float(), ref_moe,
                         atol=5e-1, rtol=5e-1)

    @unittest.skipUnless(__import__('os').environ.get("ZENTORCH_ENABLE_CHECKS"),
                         "Set ZENTORCH_ENABLE_CHECKS=1 before running")
    @torch.inference_mode()
    def test_int8_missing_scales(self):
        """Int8 weights without scales raises RuntimeError."""
        num_experts, M, K, N = 4, 8, 32, 32
        inputs = [torch.randn(M, K) for i in range(num_experts)]
        w13_int8 = [torch.randint(-128, 128, (N, K), dtype=torch.int8)
                    for i in range(num_experts)]
        w13_bias = [None] * num_experts
        gemm_outputs = [torch.empty(M, N) for i in range(num_experts)]
        with self.assertRaisesRegex(RuntimeError, "weight_scales"):
            torch.ops.zentorch.zentorch_group_matmul.out(
                gemm_outputs, inputs, w13_int8, [], None, None, None,
                "none", w13_bias, [], [None] * num_experts, [])

    @parameterized.expand(product(
        supported_dtypes,
        GROUP_MATMUL_CONFIGS["num_experts"],
        GROUP_MATMUL_CONFIGS["M"],
        GROUP_MATMUL_CONFIGS["K"],
        GROUP_MATMUL_CONFIGS["D"],
        GROUP_MATMUL_CONFIGS["K_out"],
    ))
    @torch.inference_mode()
    def test_empty_gemm_outputs_fused_w2(self, dtype_str, num_experts, M, K, D, K_out):
        """Fused w2 with gemm_outputs=[] — backend allocates dst internally."""
        torch.manual_seed(42)
        dtype = self.data.get_torch_type(dtype_str)
        N = 2 * D
        activation = "silu"
        w13_bias = [None] * num_experts
        inputs = [torch.randn(M, K, dtype=dtype) for i in range(num_experts)]
        w13_weights = [torch.randn(N, K, dtype=dtype) for i in range(num_experts)]
        w2_weights = [torch.randn(K_out, D, dtype=dtype) for i in range(num_experts)]
        w2_bias = [torch.randn(K_out, dtype=dtype) for i in range(num_experts)]

        torch.ops.zentorch.zentorch_group_matmul.out(
            [], inputs, w13_weights, w2_weights, None, None, None,
            activation, w13_bias, w2_bias, [], [])

    @parameterized.expand(product(
        supported_dtypes,
        GROUP_MATMUL_CONFIGS["num_experts"],
        GROUP_MATMUL_CONFIGS["K"],
        GROUP_MATMUL_CONFIGS["D"],
        GROUP_MATMUL_CONFIGS["topk"],
        GROUP_MATMUL_CONFIGS["num_tokens"],
    ))
    @torch.inference_mode()
    def test_fused_moe_pipeline(self, dtype_str, num_experts, K, D, topk, num_tokens):
        """Full pipeline: w13 → activation → w2 (with bias) → MoE reduce, bf16.

        Two output paths are verified per config:
          - Low-level: zentorch_group_matmul.out with inline MoE weighted-reduce
          - High-level: zentorch_fused_moe (token grouping + W13 + act + W2 +
            weighted reduce in a single op call)

        ZenDNN reuses the input buffers for w2 output, so row_ptrs must
        point into the input tensors (which will hold the down projection
        results after the kernel completes).
        Constraint: K_out must equal K (buffer reuse).
        """
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")
        if not zentorch._C.is_bf16_supported():
            self.skipTest("BF16 not supported")

        torch.manual_seed(42)
        K_out = K
        N = 2 * D
        activation = "silu"
        dtype = self.data.get_torch_type(dtype_str)

        w13_bias = [None] * num_experts
        w13_weights = [torch.randn(N, K, dtype=dtype) for i in range(num_experts)]
        w2_weights = [torch.randn(K_out, D, dtype=dtype) for i in range(num_experts)]
        w2_bias = [torch.randn(K_out, dtype=dtype) for i in range(num_experts)]

        topk_indices = torch.randint(0, num_experts, (num_tokens, topk))
        topk_weights_t = torch.rand(num_tokens, topk)
        hidden_states = torch.randn(num_tokens, K, dtype=dtype)

        inputs, unused_outputs, unused_ptrs = self._scatter_and_build_row_ptrs(
            hidden_states, topk_indices, num_experts, K, K_out)

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
            gate_up_outputs, inputs, w13_weights, w2_weights,
            moe_reduce_output, topk_weights_t, row_ptrs_into_inputs,
            activation, w13_bias, w2_bias, [], [])

        self.assertEqual(moe_reduce_output.float(), ref_moe, **TOLERANCES["fused_bf16"])

        # --- Output 2: high-level zentorch_fused_moe op ---
        w13_3d = torch.stack(w13_weights, dim=0)
        w2_3d = torch.stack(w2_weights, dim=0)
        w2_bias_3d = torch.stack(w2_bias, dim=0)

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
