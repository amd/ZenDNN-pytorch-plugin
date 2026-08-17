# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
import unittest
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: E402
    dynamic_quant_dequant_per_token,
    GROUP_MATMUL_DA8W4_GROUP_SIZE_VALUES,
    GROUP_MATMUL_DA8W4_HIDDEN_VALUES,
    GROUP_MATMUL_DA8W4_INTER_VALUES,
    GroupMatmulTestCase,
    has_zentorch,
    run_tests,
)

# Element-wise atol/rtol (mirrors test_group_matmul.py).
TOLERANCES = {
    "da8w4_fused": {"atol": 5e-2, "rtol": 5e-2},
}

# The tensor_group_matmul_strategy only builds the packed-s4 data when
# these contraction dimensions are supplied to the test function.
DA8W4_SHAPES = {
    "hidden_list": GROUP_MATMUL_DA8W4_HIDDEN_VALUES,
    "inter_list": GROUP_MATMUL_DA8W4_INTER_VALUES,
    "group_size_list": GROUP_MATMUL_DA8W4_GROUP_SIZE_VALUES,
}


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_FusedMoEDA8W4(GroupMatmulTestCase):
    """DA8W4 regime of the unified ``zentorch_fused_moe`` op.

    Shapes / packed s4 weights / routing come from the shared quantized-MoE
    data built by ``GroupMatmulTestCase.tensor_group_matmul_strategy`` (see
    ``build_quant_moe_data``); each hypothesis example draws fresh dims. The
    per-expert weight lists are stacked into the ``[E, ...]`` layout the fused
    op expects.
    """

    def setUp(self):
        super().setUp()
        torch.ops.zentorch.zentorch_flush_moe_weight_cache()

    @staticmethod
    def _stacked(quant_moe_data, torch_dtype, with_bias):
        """Stack the per-expert lists into the fused-op ``[E, ...]`` layout."""
        w13_packed = torch.stack(quant_moe_data["w13_packed"], dim=0).contiguous()
        w2_packed = torch.stack(quant_moe_data["w2_packed"], dim=0).contiguous()
        w13_scale = torch.stack(quant_moe_data["w13_scales"], dim=0).contiguous()
        w2_scale = torch.stack(quant_moe_data["w2_scales"], dim=0).contiguous()
        w13_bias = w2_bias = None
        if with_bias:
            w13_bias = (
                torch.stack(quant_moe_data["w13_bias"], dim=0)
                .to(torch_dtype)
                .contiguous()
            )
            w2_bias = (
                torch.stack(quant_moe_data["w2_bias"], dim=0)
                .to(torch_dtype)
                .contiguous()
            )
        return w13_packed, w2_packed, w13_scale, w2_scale, w13_bias, w2_bias

    def _reference(self, quant_moe_data, with_bias):
        """Pure-PyTorch DA8W4 MoE reference (dequant weight + dynamic s8 act)."""
        x = quant_moe_data["hidden_states"]
        topk_weights = quant_moe_data["topk_weights"]
        topk_id = quant_moe_data["topk_indices"]
        dq_w13, dq_w2 = quant_moe_data["dq_w13"], quant_moe_data["dq_w2"]
        w13_bias = quant_moe_data["w13_bias"] if with_bias else None
        w2_bias = quant_moe_data["w2_bias"] if with_bias else None
        num_tokens, hidden = x.shape
        top_k = topk_id.shape[1]

        out = torch.zeros(num_tokens, hidden, dtype=torch.float32)
        for t in range(num_tokens):
            row = x[t : t + 1].float()  # [1, H]
            for k in range(top_k):
                e = int(topk_id[t, k].item())
                xq = dynamic_quant_dequant_per_token(row)  # [1, H]
                gate_up = xq @ dq_w13[e].t()  # [1, 2I]
                if w13_bias is not None:
                    gate_up = gate_up + w13_bias[e].float()
                gate, up = gate_up.chunk(2, dim=-1)
                h = torch.nn.functional.silu(gate) * up  # [1, I]
                hq = dynamic_quant_dequant_per_token(h)  # [1, I]
                expert_out = hq @ dq_w2[e].t()  # [1, H]
                if w2_bias is not None:
                    expert_out = expert_out + w2_bias[e].float()
                out[t] += float(topk_weights[t, k].item()) * expert_out[0]
        return out

    def _run_op(self, quant_moe_data, torch_dtype, with_bias):
        (
            w13_packed, w2_packed, w13_scale, w2_scale, w13_bias, w2_bias,
        ) = self._stacked(quant_moe_data, torch_dtype, with_bias)
        x_c = quant_moe_data["hidden_states"].contiguous()
        # NaN-poisoned, not zeroed: the reduce post-op must write every element,
        # so anything it misses surfaces as a mismatch instead of a stray zero.
        output = torch.full_like(x_c, float("nan"))
        torch.ops.zentorch.zentorch_fused_moe(
            output,
            x_c,
            w13_packed,  # w13 (packed s4)
            w2_packed,  # w2  (packed s4)
            w13_bias,  # w13_bias [E, 2*I] bf16 or None
            w2_bias,  # w2_bias  [E, H]   bf16 or None
            quant_moe_data["topk_weights"],
            quant_moe_data["topk_indices"].to(torch.int32),
            False,  # skip_weighted (apply_router_weight_on_input)
            "silu",
            w13_scale,  # w13_scales (per-group)
            w2_scale,  # w2_scales  (per-group)
        )
        return output

    def _assert_close(self, op_out, ref_out, label):
        self.assertEqual(op_out, ref_out, msg=label, **TOLERANCES["da8w4_fused"])

    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=["bfloat16"],
        **DA8W4_SHAPES,
    )
    @torch.inference_mode()
    def test_fused_moe_da8w4_accuracy(self, dtype, with_bias):
        """DA8W4 fused MoE accuracy. ``with_bias`` covers the path taken when a
        checkpoint carries per-expert w13/w2 biases."""
        torch_dtype = self.data.get_torch_type(dtype)
        quant_moe_data = self.data.group_matmul_quant_moe_data
        op_out = self._run_op(quant_moe_data, torch_dtype, with_bias).float()
        ref_out = self._reference(quant_moe_data, with_bias)
        self._assert_close(
            op_out, ref_out, f"fused MoE DA8W4 (with_bias={with_bias})"
        )

    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=["bfloat16"],
        **DA8W4_SHAPES,
    )
    @torch.inference_mode()
    def test_fused_moe_da8w4_requires_bf16(self, dtype, with_bias):
        """DA8W4 packed (int32) weights require a bf16 activation; a float32
        input must fail fast (the DA8W4 kernel rejects f32)."""
        torch_dtype = self.data.get_torch_type(dtype)
        quant_moe_data = self.data.group_matmul_quant_moe_data
        (
            w13_packed, w2_packed, w13_scale, w2_scale, w13_bias, w2_bias,
        ) = self._stacked(quant_moe_data, torch_dtype, with_bias)
        # Wrong dtype for DA8W4.
        x_f32 = quant_moe_data["hidden_states"].float().contiguous()
        output = torch.zeros_like(x_f32)
        with self.assertRaises(RuntimeError):
            torch.ops.zentorch.zentorch_fused_moe(
                output,
                x_f32,
                w13_packed,
                w2_packed,
                w13_bias,
                w2_bias,
                quant_moe_data["topk_weights"],
                quant_moe_data["topk_indices"].to(torch.int32),
                False,  # skip_weighted
                "silu",
                w13_scale,
                w2_scale,
            )


if __name__ == "__main__":
    run_tests()
