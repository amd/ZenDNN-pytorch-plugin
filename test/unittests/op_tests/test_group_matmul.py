# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import os
import unittest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: E402
    GroupMatmulTestCase,
    GROUP_MATMUL_INT8_K_VALUES,
    GROUP_MATMUL_INT8_GATED_K_VALUES,
    has_zentorch,
    run_tests,
    supported_dtypes,
)
from zentorch._utils import (  # noqa: E402
    _SUPPORTED_MOE_ACTIVATIONS as SUPPORTED_ACTIVATIONS,
)

# The dynamic-int8 group_matmul path quantizes activations from bf16/fp32 only
# (see cpp/GroupMatmul.cpp: "Dynamic int8: bf16/fp32 input x s8 weight"), so
# fp16 is not supported on the int8 paths and is excluded from those tests.
supported_dtypes_int8 = [d for d in supported_dtypes if (d != "float16" and d != "float32")]

# Tolerance per dtype
TOLERANCES = {
    torch.float32: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 3e-2, "rtol": 3e-2},
    "fused_bf16": {"atol": 5e-1, "rtol": 5e-1},
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
}


def moe_output_buffer(num_tokens, hidden_dim, dtype):
    """NaN-filled MoE output buffer.

    The weighted-reduce post-op overwrites every element, so callers do not need
    to zero-initialize. Poisoning with NaN makes any element the
    kernel fails to write surface as a mismatch against the finite reference
    instead of silently passing on a leftover value.
    """
    return torch.full((num_tokens, hidden_dim), float("nan"), dtype=dtype)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GroupMatmul(GroupMatmulTestCase):
    """Hypothesis-based tests for zentorch_group_matmul.out."""

    # ------------------------------------------------------------------
    # Reference helpers
    # ------------------------------------------------------------------

    def _reference_weighted_reduce(
        self, expert_outputs, topk_weights, topk_indices, num_tokens, topk
    ):
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
                output[t] += (
                    topk_weights[t, k].item() * expert_outputs[expert_id][row_j]
                )
        return output

    def _scatter_and_build_row_ptrs(
        self, hidden_states, topk_indices, num_experts, K, N
    ):
        """Scatter tokens to experts, pre-allocate outputs, build row_ptrs."""
        num_tokens = hidden_states.shape[0]
        topk = topk_indices.shape[1]
        expert_token_counts = [
            (topk_indices == e).sum().item() for e in range(num_experts)
        ]
        inputs = [
            torch.zeros(int(expert_token_counts[e]), K, dtype=hidden_states.dtype)
            for e in range(num_experts)
        ]
        gemm_outputs = [
            torch.empty(int(expert_token_counts[e]), N, dtype=hidden_states.dtype)
            for e in range(num_experts)
        ]
        row_ptrs = torch.zeros(num_tokens * topk, dtype=torch.int64)
        expert_cursor = [0] * num_experts
        for t in range(num_tokens):
            for k in range(topk):
                expert_id = topk_indices[t, k].item()
                row_j = expert_cursor[expert_id]
                inputs[expert_id][row_j] = hidden_states[t]
                row_ptrs[t * topk + k] = (
                    gemm_outputs[expert_id].data_ptr()
                    + row_j
                    * gemm_outputs[expert_id].stride(0)
                    * gemm_outputs[expert_id].element_size()
                )
                expert_cursor[expert_id] += 1
        return inputs, gemm_outputs, row_ptrs

    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes,
    )
    @torch.inference_mode()
    def test_plain_gemm(self, dtype):
        """Plain parallel GEMM."""
        torch_dtype = self.data.get_torch_type(dtype)
        num_experts = self.data.num_experts
        M = self.data.M
        N = self.data.N
        w13_bias = self.data.w13_bias_none
        inputs = self.data.inputs
        w13 = self.data.w13_weights

        ref = self._reference_expert_outputs(inputs, w13, w13_bias)

        gemm_outputs = [
            torch.empty(M, N, dtype=torch_dtype) for i in range(num_experts)
        ]
        torch.ops.zentorch.zentorch_group_matmul.out(
            gemm_outputs,
            inputs,
            w13,
            [],
            None,
            None,
            None,
            "none",
            w13_bias,
            [],
            [],
            [],
            [],
        )

        tol = TOLERANCES[torch_dtype]
        for i in range(num_experts):
            self.assertEqual(gemm_outputs[i], ref[i], **tol)

    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes,
    )
    @torch.inference_mode()
    def test_moe_weighted_reduce(self, dtype):
        """GEMM + MoE weighted-reduce."""
        torch_dtype = self.data.get_torch_type(dtype)
        num_experts = self.data.num_experts
        K = self.data.K
        N = self.data.N
        topk = self.data.topk
        num_tokens = self.data.num_tokens
        w13_bias = self.data.w13_bias_none
        topk_indices = self.data.topk_indices
        topk_weights = self.data.topk_weights_routing
        hidden_states = self.data.hidden_states

        inputs, moe_gemm_outputs, row_ptrs = self._scatter_and_build_row_ptrs(
            hidden_states, topk_indices, num_experts, K, N
        )
        w13 = self.data.w13_weights
        moe_output = moe_output_buffer(num_tokens, N, torch_dtype)

        is_reduced_precision = torch_dtype in (torch.bfloat16, torch.float16)
        ref_experts = self._reference_expert_outputs(
            inputs, w13, w13_bias, compute_in_fp32=is_reduced_precision
        )
        ref_moe = self._reference_weighted_reduce(
            ref_experts, topk_weights, topk_indices, num_tokens, topk
        )

        torch.ops.zentorch.zentorch_group_matmul.out(
            moe_gemm_outputs,
            inputs,
            w13,
            [],
            moe_output,
            topk_weights,
            row_ptrs,
            "none",
            w13_bias,
            [],
            [],
            [],
            [],
        )

        tol = TOLERANCES[torch_dtype]
        actual = moe_output.float() if is_reduced_precision else moe_output
        self.assertEqual(actual, ref_moe, **tol)

    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes,
    )
    @torch.inference_mode()
    def test_gated_activations(self, dtype):
        """GEMM + gated activation across activation types."""
        torch_dtype = self.data.get_torch_type(dtype)
        num_experts = self.data.num_experts
        M = self.data.M
        D = self.data.D
        N = 2 * D
        inputs = self.data.inputs
        w13 = self.data.w13_weights_gated
        for activation in SUPPORTED_ACTIVATIONS:
            w13_bias = self.data.w13_bias_none

            ref = self._reference_expert_outputs(
                inputs, w13, w13_bias, activation=activation
            )

            gemm_outputs = [
                torch.empty(M, N, dtype=torch_dtype) for i in range(num_experts)
            ]
            torch.ops.zentorch.zentorch_group_matmul.out(
                gemm_outputs,
                inputs,
                w13,
                [],
                None,
                None,
                None,
                activation,
                w13_bias,
                [],
                [],
                [],
                [],
            )

            tol = TOLERANCES[torch_dtype]
            for i in range(num_experts):
                self.assertEqual(gemm_outputs[i][:, :D], ref[i], **tol)

    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes,
    )
    @torch.inference_mode()
    def test_unsupported_activation(self, dtype):
        """Unsupported activation string raises RuntimeError."""
        torch_dtype = self.data.get_torch_type(dtype)
        num_experts = self.data.num_experts
        M = self.data.M
        N = self.data.N
        inputs = self.data.inputs
        w13 = self.data.w13_weights
        w13_bias = self.data.w13_bias_none
        gemm_outputs = [
            torch.empty(M, N, dtype=torch_dtype) for i in range(num_experts)
        ]

        for bad_activation in ["relu", "tanh"]:
            with self.assertRaisesRegex(RuntimeError, "unsupported activation"):
                torch.ops.zentorch.zentorch_group_matmul.out(
                    gemm_outputs,
                    inputs,
                    w13,
                    [],
                    None,
                    None,
                    None,
                    bad_activation,
                    w13_bias,
                    [],
                    [],
                    [],
                    [],
                )

    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes_int8,
        k_list=GROUP_MATMUL_INT8_K_VALUES,
    )
    @torch.inference_mode()
    def test_int8_w13(self, dtype):
        """Dynamic int8: inputs x w13 int8 weights with per-channel scales."""
        torch_dtype = self.data.get_torch_type(dtype)
        num_experts = self.data.num_experts
        M = self.data.M
        N = self.data.N
        inputs = self.data.inputs
        w13_int8 = self.data.w13_weights_int8
        w13_scales = self.data.w13_scales
        w13_bias = self.data.w13_bias_none

        ref = []
        for i in range(num_experts):
            w_deq = w13_int8[i].float() * w13_scales[i].unsqueeze(1)
            ref.append(torch.nn.functional.linear(inputs[i].float(), w_deq, None))

        gemm_outputs = [
            torch.empty(M, N, dtype=torch_dtype) for i in range(num_experts)
        ]
        torch.ops.zentorch.zentorch_group_matmul.out(
            gemm_outputs,
            inputs,
            w13_int8,
            [],
            None,
            None,
            None,
            "none",
            w13_bias,
            [],
            w13_scales,
            [],
            [],
        )

        for i in range(num_experts):
            self.assertEqual(gemm_outputs[i].float(), ref[i], atol=2e-1, rtol=2e-1)

    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes_int8,
        k_list=GROUP_MATMUL_INT8_K_VALUES,
    )
    @torch.inference_mode()
    def test_int8_w13_and_w2_single_pass(self, dtype):
        """Dynamic int8 w13 + int8 w2 with per-channel scales (single-call path).

        Three sub-tests:

        1. ``zentorch_group_matmul.out`` — no activation, no MoE reduce
           (square DA8W8, bf16 inputs, fused w2 reuses input buffers).
        2. ``zentorch_group_matmul.out`` — silu gated activation + MoE
           weighted reduce (bf16 inputs, fused w2).
        3. ``zentorch_fused_moe`` — silu + MoE weighted reduce. Unique-token
           s8 pre-quant, then one fused ``group_matmul_direct`` (W13 + act +
           W2 + reduce).

        Sub-tests 1–2 require K == K_out == N for fused w2 buffer reuse.
        """
        num_experts = self.data.num_experts
        K = self.data.K
        topk = self.data.topk
        num_tokens = self.data.num_tokens

        torch_dtype = self.data.get_torch_type(dtype)
        K_out = K
        activation = "none"

        def dynamic_quant_matmul(src, w_int8, w_scales, bias=None, out_dtype=None):
            src_fp = src.float()
            src_scale = src_fp.abs().amax(dim=1).clamp(min=1e-12) / 127.0
            src_q = (src_fp / src_scale.unsqueeze(1)).round().clamp(-128, 127)
            acc = src_q @ w_int8.float().T
            result = acc * (src_scale.unsqueeze(1) * w_scales.unsqueeze(0))
            if bias is not None:
                result = result + bias.float()
            # Round to the kernel's intermediate dtype: it writes each GEMM
            # result to a working-dtype (bf16/f32) buffer before the next op
            # re-reads it. A full-fp32 reference would differ by ~1 int8 LSB
            # near a quant boundary.
            if out_dtype is not None:
                result = result.to(out_dtype).float()
            return result

        # --- Sub-test 1: group_matmul, no activation, no MoE reduce ---
        w13_int8 = self.data.w13_weights_int8_square
        w13_scales = self.data.w13_scales_square
        w13_bias = self.data.w13_bias_none
        w2_int8 = self.data.w2_weights_int8_square
        w2_scales = self.data.w2_scales_square
        w2_bias = self.data.w2_bias_gated

        # MoE routing tensors from the strategy.
        hidden_states = self.data.hidden_states
        topk_indices = self.data.topk_indices
        topk_weights_t = self.data.topk_weights_routing

        inputs, unused_outputs, unused_ptrs = self._scatter_and_build_row_ptrs(
            hidden_states, topk_indices, num_experts, K, K_out
        )

        ref = []
        for i in range(num_experts):
            r = dynamic_quant_matmul(
                inputs[i], w13_int8[i], w13_scales[i], out_dtype=torch_dtype
            )
            r = dynamic_quant_matmul(
                r, w2_int8[i], w2_scales[i], bias=w2_bias[i], out_dtype=torch_dtype
            )
            ref.append(r)

        torch.ops.zentorch.zentorch_group_matmul.out(
            [],
            inputs,
            w13_int8,
            w2_int8,
            None,
            None,
            None,
            activation,
            w13_bias,
            w2_bias,
            w13_scales,
            w2_scales,
            [],
        )

        for i in range(num_experts):
            self.assertEqual(inputs[i].float(), ref[i], atol=5e-1, rtol=5e-1)

        # --- Sub-test 2: group_matmul, silu activation + MoE weighted reduce ---
        w13_act_int8 = self.data.w13_weights_int8_gated
        w13_act_scales = self.data.w13_scales_gated
        w13_act_bias = self.data.w13_bias_gated
        w2_act_int8 = self.data.w2_weights_int8_gated
        w2_act_scales = self.data.w2_scales_gated
        w2_act_bias = self.data.w2_bias_gated

        inputs_act_moe, gemm_outputs_act_moe, row_ptrs_act_moe = (
            self._scatter_and_build_row_ptrs(
                hidden_states, topk_indices, num_experts, K, K_out
            )
        )

        # Build row_ptrs pointing into inputs (fused w2 writes back there)
        row_ptrs_act_moe = torch.zeros(num_tokens * topk, dtype=torch.int64)
        per_expert_row_act = [0] * num_experts
        for token_idx in range(num_tokens):
            for topk_idx in range(topk):
                expert_id = topk_indices[token_idx, topk_idx].item()
                row_in_expert = per_expert_row_act[expert_id]
                row_ptrs_act_moe[token_idx * topk + topk_idx] = (
                    inputs_act_moe[expert_id].data_ptr()
                    + row_in_expert
                    * inputs_act_moe[expert_id].stride(0)
                    * inputs_act_moe[expert_id].element_size()
                )
                per_expert_row_act[expert_id] += 1

        ref_act_moe = []
        for i in range(num_experts):
            r = dynamic_quant_matmul(
                inputs_act_moe[i],
                w13_act_int8[i],
                w13_act_scales[i],
                bias=w13_act_bias[i],
                out_dtype=torch_dtype,
            )
            r = self._apply_gated_activation(r, "silu")
            # Kernel stores the activation output back into the working-dtype
            # Op1 buffer before Op2 re-quantizes it; round to match.
            r = r.to(torch_dtype).float()
            r = dynamic_quant_matmul(
                r,
                w2_act_int8[i],
                w2_act_scales[i],
                bias=w2_act_bias[i],
                out_dtype=torch_dtype,
            )
            ref_act_moe.append(r)
        ref_moe_act = self._reference_weighted_reduce(
            ref_act_moe, topk_weights_t, topk_indices, num_tokens, topk
        )

        moe_reduce_act = moe_output_buffer(num_tokens, K_out, torch_dtype)

        torch.ops.zentorch.zentorch_group_matmul.out(
            [],
            inputs_act_moe,
            w13_act_int8,
            w2_act_int8,
            moe_reduce_act,
            topk_weights_t,
            row_ptrs_act_moe,
            "silu",
            w13_act_bias,
            w2_act_bias,
            w13_act_scales,
            w2_act_scales,
            [],
        )

        self.assertEqual(moe_reduce_act.shape, (num_tokens, K_out))
        self.assertEqual(
            moe_reduce_act.float(), ref_moe_act, **TOLERANCES["fused_bf16"]
        )

        # --- Sub-test 3: fused_moe, unique-token s8 + silu + MoE reduce ---
        w13_fmoe_int8 = self.data.w13_weights_int8_gated
        w13_fmoe_scales = self.data.w13_scales_gated
        w2_fmoe_int8 = self.data.w2_weights_int8_gated
        w2_fmoe_scales = self.data.w2_scales_gated
        w2_fmoe_bias = self.data.w2_bias_gated

        inputs_fmoe, unused_outputs_fmoe, unused_ptrs_fmoe = (
            self._scatter_and_build_row_ptrs(
                hidden_states, topk_indices, num_experts, K, K_out
            )
        )

        ref_fmoe = []
        for i in range(num_experts):
            r = dynamic_quant_matmul(
                inputs_fmoe[i],
                w13_fmoe_int8[i],
                w13_fmoe_scales[i],
                out_dtype=torch_dtype,
            )
            r = self._apply_gated_activation(r, "silu")
            r = r.to(torch_dtype).float()
            r = dynamic_quant_matmul(
                r,
                w2_fmoe_int8[i],
                w2_fmoe_scales[i],
                bias=w2_fmoe_bias[i],
                out_dtype=torch_dtype,
            )
            ref_fmoe.append(r)
        ref_moe_fmoe = self._reference_weighted_reduce(
            ref_fmoe, topk_weights_t, topk_indices, num_tokens, topk
        )

        w13_3d = torch.stack(w13_fmoe_int8, dim=0)
        w2_3d = torch.stack(w2_fmoe_int8, dim=0)
        w13_scales_3d = torch.stack(w13_fmoe_scales, dim=0)
        w2_scales_3d = torch.stack(w2_fmoe_scales, dim=0)
        w2_bias_3d = torch.stack(w2_fmoe_bias, dim=0)

        fused_moe_output = moe_output_buffer(num_tokens, K_out, torch_dtype)
        torch.ops.zentorch.zentorch_fused_moe(
            fused_moe_output,
            hidden_states,
            w13_3d,
            w2_3d,
            None,  # no w13_bias
            w2_bias_3d,
            topk_weights_t,
            topk_indices.to(torch.int32),
            False,  # skip_weighted
            "silu",
            w13_scales_3d,
            w2_scales_3d,
        )

        self.assertEqual(fused_moe_output.shape, (num_tokens, K_out))
        self.assertEqual(
            fused_moe_output.float(), ref_moe_fmoe, **TOLERANCES["fused_bf16"]
        )

    @unittest.skipUnless(
        os.environ.get("ZENTORCH_TWO_PASS") == "1",
        "Set ZENTORCH_TWO_PASS=1 before running",
    )
    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes_int8,
        k_list=GROUP_MATMUL_INT8_GATED_K_VALUES,
    )
    @torch.inference_mode()
    def test_int8_w13_and_w2_two_pass(self, dtype):
        """Dynamic int8 w13 + int8 w2 with per-channel scales via zentorch_fused_moe.

        Exercises the full MoE chain (W13 -> gated activation -> W2 -> weighted
        reduce) for int8 weights as two ``group_matmul_direct`` calls.
        ``ZENTORCH_TWO_PASS=1`` skips unique-token pre-quant and keeps bf16
        grouping; this test requires that env so it covers the split, not the
        default fused unique-token path. For fused grouped-quant without the
        split, set ``ZENTORCH_MOE_PREQUANT=0`` (and leave ``TWO_PASS`` unset).

        Constraint: K == K_out (W2 output dim).
        """
        torch_dtype = self.data.get_torch_type(dtype)
        num_experts = self.data.num_experts
        K = self.data.K
        topk = self.data.topk
        num_tokens = self.data.num_tokens
        activation = "silu"
        K_out = K

        def dynamic_quant_matmul(src, w_int8, w_scales, bias=None, out_dtype=None):
            src_fp = src.float()
            src_scale = src_fp.abs().amax(dim=1).clamp(min=1e-12) / 127.0
            src_q = (src_fp / src_scale.unsqueeze(1)).round().clamp(-128, 127)
            acc = src_q @ w_int8.float().T
            result = acc * (src_scale.unsqueeze(1) * w_scales.unsqueeze(0))
            if bias is not None:
                result = result + bias.float()
            # Round to the kernel's intermediate dtype: it writes each GEMM
            # result to a working-dtype (bf16/f32) buffer before the next op
            # re-reads it. A full-fp32 reference would differ by ~1 int8 LSB
            # near a quant boundary.
            if out_dtype is not None:
                result = result.to(out_dtype).float()
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

        w13_int8 = self.data.w13_weights_int8_gated
        w13_scales = self.data.w13_scales_gated
        w2_int8 = self.data.w2_weights_int8_gated
        w2_scales = self.data.w2_scales_gated
        w2_bias = self.data.w2_bias_gated

        hidden_states = self.data.hidden_states
        topk_indices = self.data.topk_indices
        topk_weights_t = self.data.topk_weights_routing

        inputs, unused_outputs, unused_ptrs = self._scatter_and_build_row_ptrs(
            hidden_states, topk_indices, num_experts, K, K_out
        )

        ref_experts = []
        for i in range(num_experts):
            r = dynamic_quant_matmul(inputs[i], w13_int8[i], w13_scales[i])
            r = apply_gated_activation(r, activation)
            r = dynamic_quant_matmul(r, w2_int8[i], w2_scales[i], bias=w2_bias[i])
            ref_experts.append(r)
        ref_moe = self._reference_weighted_reduce(
            ref_experts, topk_weights_t, topk_indices, num_tokens, topk
        )

        w13_3d = torch.stack(w13_int8, dim=0)
        w2_3d = torch.stack(w2_int8, dim=0)
        w13_scales_3d = torch.stack(w13_scales, dim=0)
        w2_scales_3d = torch.stack(w2_scales, dim=0)
        w2_bias_3d = torch.stack(w2_bias, dim=0)

        fused_moe_output = moe_output_buffer(num_tokens, K_out, torch_dtype)
        torch.ops.zentorch.zentorch_fused_moe(
            fused_moe_output,
            hidden_states,
            w13_3d,
            w2_3d,
            None,  # no w13_bias
            w2_bias_3d,
            topk_weights_t,
            topk_indices.to(torch.int32),
            False,  # skip_weighted
            activation,
            w13_scales_3d,
            w2_scales_3d,
        )

        self.assertEqual(fused_moe_output.shape, (num_tokens, K_out))
        self.assertEqual(fused_moe_output.float(), ref_moe, atol=5e-1, rtol=5e-1)

    @unittest.skipUnless(
        os.environ.get("ZENTORCH_ENABLE_CHECKS"),
        "Set ZENTORCH_ENABLE_CHECKS=1 before running",
    )
    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes_int8,
        k_list=GROUP_MATMUL_INT8_K_VALUES,
    )
    @torch.inference_mode()
    def test_int8_missing_scales(self, dtype):
        """Int8 weights without scales raises RuntimeError."""
        torch_dtype = self.data.get_torch_type(dtype)
        num_experts = self.data.num_experts
        M = self.data.M
        N = self.data.N
        inputs = self.data.inputs
        w13_int8 = self.data.w13_int8_raw
        w13_bias = self.data.w13_bias_none
        gemm_outputs = [
            torch.empty(M, N, dtype=torch_dtype) for i in range(num_experts)
        ]
        with self.assertRaisesRegex(RuntimeError, "weight_scales"):
            torch.ops.zentorch.zentorch_group_matmul.out(
                gemm_outputs,
                inputs,
                w13_int8,
                [],
                None,
                None,
                None,
                "none",
                w13_bias,
                [],
                [None] * num_experts,
                [],
                [],
            )

    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes,
    )
    @torch.inference_mode()
    def test_empty_gemm_outputs_fused_w2(self, dtype):
        """Fused w2 with gemm_outputs=[] — backend allocates dst internally.

        Buffer-reuse constraint: w2 writes back into the per-expert input
        buffers ([M_e, K]), so K_out must equal K.
        """
        activation = "silu"
        w13_bias = self.data.w13_bias_none
        inputs = self.data.inputs
        w13_weights = self.data.w13_weights_gated
        w2_weights = self.data.w2_weights_gated
        w2_bias = self.data.w2_bias_gated

        torch.ops.zentorch.zentorch_group_matmul.out(
            [],
            inputs,
            w13_weights,
            w2_weights,
            None,
            None,
            None,
            activation,
            w13_bias,
            w2_bias,
            [],
            [],
            [],
        )

    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes,
    )
    @torch.inference_mode()
    def test_fused_moe_pipeline(self, dtype):
        """Full pipeline: w13 → activation → w2 (with bias) → MoE reduce, bf16.

        Two output paths are verified per config:
        - Low-level: zentorch_group_matmul.out with inline MoE weighted-reduce
        - High-level: zentorch_fused_moe (token grouping + W13 + act + W2 +
            weighted reduce in a single op call)

        ZenDNN reuses the gemm_outputs buffers for w2 output, so row_ptrs must
        point into the gemm_outputs tensors (which will hold the down projection
        results after the kernel completes).
        Constraint: K_out must equal K (buffer reuse).
        """
        torch_dtype = self.data.get_torch_type(dtype)
        num_experts = self.data.num_experts
        K = self.data.K
        D = self.data.D
        topk = self.data.topk
        num_tokens = self.data.num_tokens
        K_out = K
        N = 2 * D
        activation = "silu"

        w13_bias = self.data.w13_bias_none
        w13_weights = self.data.w13_weights_gated
        w2_weights = self.data.w2_weights_gated
        w2_bias = self.data.w2_bias_gated

        topk_indices = self.data.topk_indices
        topk_weights_t = self.data.topk_weights_routing
        hidden_states = self.data.hidden_states

        inputs, unused_outputs, unused_ptrs = self._scatter_and_build_row_ptrs(
            hidden_states, topk_indices, num_experts, K, K_out
        )

        gate_up_outputs = [
            torch.empty(t.size(0), N, dtype=torch_dtype) for t in inputs
        ]

        row_ptrs_into_gate_up = torch.zeros(num_tokens * topk, dtype=torch.int64)
        per_expert_row = [0] * num_experts
        for token_idx in range(num_tokens):
            for topk_idx in range(topk):
                expert_id = topk_indices[token_idx, topk_idx].item()
                row_in_expert = per_expert_row[expert_id]
                row_ptrs_into_gate_up[token_idx * topk + topk_idx] = (
                    gate_up_outputs[expert_id].data_ptr()
                    + row_in_expert
                    * gate_up_outputs[expert_id].stride(0)
                    * gate_up_outputs[expert_id].element_size()
                )
                per_expert_row[expert_id] += 1

        ref_down = self._reference_expert_outputs(
            inputs,
            w13_weights,
            w13_bias,
            activation=activation,
            w2_weights=w2_weights,
            w2_bias=w2_bias,
            compute_in_fp32=(torch_dtype in (torch.bfloat16, torch.float16)),
        )
        ref_moe = self._reference_weighted_reduce(
            ref_down, topk_weights_t, topk_indices, num_tokens, topk
        )

        # --- Output 1: low-level zentorch_group_matmul.out ---
        active_ids = [e for e in range(num_experts) if inputs[e].size(0) > 0]
        inactive_ids = [e for e in range(num_experts) if inputs[e].size(0) == 0]
        weight_order = active_ids + inactive_ids

        inputs_active = [inputs[e] for e in active_ids]
        w13_ordered = [w13_weights[e] for e in weight_order]
        w2_ordered = [w2_weights[e] for e in weight_order]
        w13_bias_active = [w13_bias[e] for e in active_ids]
        w2_bias_active = [w2_bias[e] for e in active_ids]

        moe_reduce_output = moe_output_buffer(num_tokens, K_out, torch_dtype)
        gate_up_outputs_active = [gate_up_outputs[e] for e in active_ids]

        torch.ops.zentorch.zentorch_group_matmul.out(
            gate_up_outputs_active,
            inputs_active,
            w13_ordered,
            w2_ordered,
            moe_reduce_output,
            topk_weights_t,
            row_ptrs_into_gate_up,
            activation,
            w13_bias_active,
            w2_bias_active,
            [],
            [],
            [],
        )

        self.assertEqual(moe_reduce_output.float(), ref_moe, **TOLERANCES["fused_bf16"])

        # --- Output 2: high-level zentorch_fused_moe op ---
        w13_3d = torch.stack(w13_weights, dim=0)
        w2_3d = torch.stack(w2_weights, dim=0)
        w2_bias_3d = torch.stack(w2_bias, dim=0)

        fused_moe_output = moe_output_buffer(num_tokens, K_out, torch_dtype)
        torch.ops.zentorch.zentorch_fused_moe(
            fused_moe_output,
            hidden_states,
            w13_3d,
            w2_3d,
            None,  # no w13_bias
            w2_bias_3d,
            topk_weights_t,
            topk_indices.to(torch.int32),
            False,  # skip_weighted
            activation,
        )

        self.assertEqual(fused_moe_output.shape, (num_tokens, K_out))
        self.assertEqual(fused_moe_output.float(), ref_moe, **TOLERANCES["fused_bf16"])


if __name__ == "__main__":
    run_tests()
