# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: E402
    GROUP_MATMUL_INT8_K_VALUES,
    GROUP_MATMUL_DA8W4_GROUP_SIZE_VALUES,
    GROUP_MATMUL_DA8W4_HIDDEN_VALUES,
    GROUP_MATMUL_DA8W4_INTER_VALUES,
    GroupMatmulTestCase,
    dynamic_quant_dequant_per_token,
    has_zentorch,
    run_tests,
    supported_dtypes,
    _clear_weight_cache,
)

# Dynamic-int8 fused FFN quantizes bf16 activations; fp16/fp32 are excluded.
supported_dtypes_int8 = [
    d for d in supported_dtypes if (d != "float16" and d != "float32")
]

FFN_ACTIVATIONS = []
if has_zentorch:
    from zentorch._utils import _SUPPORTED_MOE_ACTIVATIONS
    FFN_ACTIVATIONS = _SUPPORTED_MOE_ACTIVATIONS

# Tolerance per dtype (mirrors test_group_matmul.py). The fused W13 -> act -> W2
# chain accumulates error at reduced precision, so bf16/fp16 compare against an
# fp32 reference through the looser "fused_bf16" band while fp32 stays tight.
TOLERANCES = {
    torch.float32: {"atol": 1e-3, "rtol": 1e-3},
    torch.bfloat16: {"atol": 3e-2, "rtol": 3e-2},
    "fused_bf16": {"atol": 5e-1, "rtol": 5e-1},
    torch.float16: {"atol": 1e-2, "rtol": 1e-2},
    "da8w4_fused": {"atol": 5e-2, "rtol": 5e-2},
}

DA8W4_SHAPES = {
    "hidden_list": GROUP_MATMUL_DA8W4_HIDDEN_VALUES,
    "inter_list": GROUP_MATMUL_DA8W4_INTER_VALUES,
    "group_size_list": GROUP_MATMUL_DA8W4_GROUP_SIZE_VALUES,
}


def ffn_output_buffer(x):
    """ NaN-filled output buffer shaped like ``x`` """
    return torch.full_like(x, float("nan"))


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_FusedFFNConcat(GroupMatmulTestCase):
    """Single-expert (non-MoE) ``zentorch_fused_ffn_concat.out`` tests."""

    # ------------------------------------------------------------------
    # Reference helpers (mirror test_group_matmul.py).
    # ------------------------------------------------------------------

    def _reference_ffn(
        self, x, w13, w2, w13_bias, w2_bias, activation, compute_in_fp32
    ):
        x_s = [x]
        w13_s = [w13]
        w2_s = [w2]
        w13_bias_s = [w13_bias]
        w2_bias_s = [w2_bias]
        return self._reference_expert_outputs(x_s, w13_s, w13_bias_s, activation, w2_s, w2_bias_s, compute_in_fp32)[0]

    def _single_expert_weights(self, with_bias):
        """First-expert gated weights as the single-expert FFN's W13 / W2.

        ``w13_weights_gated[e]`` is ``[2*I, H]`` and ``w2_weights_gated[e]`` is
        ``[H, I]`` -- exactly the layout ``fused_ffn_concat`` expects.
        """
        w13 = self.data.w13_weights_gated[0]
        w2 = self.data.w2_weights_gated[0]
        w13_bias = self.data.w13_bias_gated[0] if with_bias else None
        w2_bias = self.data.w2_bias_gated[0] if with_bias else None
        return w13, w2, w13_bias, w2_bias

    def _run_op(self, x, w13, w2, w13_bias, w2_bias, activation):
        """Run the out-variant op; return (output, input_after_call)."""
        output = ffn_output_buffer(x)
        # Clone the input so an accidental in-place write can be detected.
        op_input = x.clone()
        torch.ops.zentorch.zentorch_fused_ffn_concat.out(
            output,
            op_input,
            w13,
            w2,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
            activation=activation,
        )
        return output, op_input

    # ------------------------------------------------------------------
    # Accuracy: 2D input, all supported gated activations, +/- bias.
    # ------------------------------------------------------------------
    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes,
    )
    @torch.inference_mode()
    def test_fused_ffn_accuracy_2d(self, dtype, with_bias):
        torch_dtype = self.data.get_torch_type(dtype)
        is_reduced_precision = torch_dtype in (torch.bfloat16, torch.float16)
        tol = (
            TOLERANCES["fused_bf16"]
            if is_reduced_precision
            else TOLERANCES[torch_dtype]
        )

        x = self.data.inputs[0]  # [M, H]
        w13, w2, w13_bias, w2_bias = self._single_expert_weights(with_bias)

        for activation in FFN_ACTIVATIONS:
            _clear_weight_cache()
            ref = self._reference_ffn(
                x, w13, w2, w13_bias, w2_bias, activation,
                compute_in_fp32=is_reduced_precision,
            )
            output, _ = self._run_op(
                x, w13, w2, w13_bias, w2_bias, activation
            )

            self.assertEqual(output.shape, ref.shape)
            self.assertFalse(
                torch.isnan(output).any(),
                f"output left uninitialized (activation={activation})",
            )
            actual = output.float() if is_reduced_precision else output
            self.assertEqual(actual, ref, **tol)

    # ------------------------------------------------------------------
    # Accuracy: 3D input [B, S, H] -- the op flattens to 2D internally.
    # ------------------------------------------------------------------
    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes,
    )
    @torch.inference_mode()
    def test_fused_ffn_accuracy_3d(self, dtype, with_bias):
        torch_dtype = self.data.get_torch_type(dtype)
        is_reduced_precision = torch_dtype in (torch.bfloat16, torch.float16)
        tol = (
            TOLERANCES["fused_bf16"]
            if is_reduced_precision
            else TOLERANCES[torch_dtype]
        )

        # Reshape the [M, H] expert input to a 3D [1, M, H] activation.
        x = self.data.inputs[0].unsqueeze(0).contiguous()  # [1, M, H]
        w13, w2, w13_bias, w2_bias = self._single_expert_weights(with_bias)
        activation = "silu"

        ref = self._reference_ffn(
            x, w13, w2, w13_bias, w2_bias, activation,
            compute_in_fp32=is_reduced_precision,
        )
        output, _ = self._run_op(x, w13, w2, w13_bias, w2_bias, activation)

        self.assertEqual(output.shape, ref.shape)
        self.assertEqual(output.dim(), 3)
        self.assertFalse(torch.isnan(output).any())
        actual = output.float() if is_reduced_precision else output
        self.assertEqual(actual, ref, **tol)

    # ------------------------------------------------------------------
    # Unsupported activation strings must raise.
    # ------------------------------------------------------------------
    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes,
    )
    @torch.inference_mode()
    def test_unsupported_activation(self, dtype):
        x = self.data.inputs[0]
        w13, w2, _, _ = self._single_expert_weights(with_bias=False)
        output = ffn_output_buffer(x)

        for bad_activation in ("relu", "tanh"):
            with self.assertRaisesRegex(RuntimeError, "unsupported activation"):
                torch.ops.zentorch.zentorch_fused_ffn_concat.out(
                    output,
                    x.clone(),
                    w13,
                    w2,
                    activation=bad_activation,
                )

    # ------------------------------------------------------------------
    # DA8W8: int8 weights + per-channel scales, bf16 activations.
    # ------------------------------------------------------------------
    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=supported_dtypes_int8,
        k_list=GROUP_MATMUL_INT8_K_VALUES,
    )
    @torch.inference_mode()
    def test_fused_ffn_int8_da8w8(self, dtype, with_bias):
        torch_dtype = self.data.get_torch_type(dtype)
        x = self.data.inputs[0]
        w13 = self.data.w13_weights_int8_gated[0]
        w2 = self.data.w2_weights_int8_gated[0]
        w13_scale = self.data.w13_scales_gated[0]
        w2_scale = self.data.w2_scales_gated[0]
        w13_bias = self.data.w13_bias_gated[0] if with_bias else None
        w2_bias = self.data.w2_bias_gated[0] if with_bias else None
        activation = "silu"

        def dynamic_quant_matmul(src, w_int8, w_scales, bias=None):
            src_fp = src.float()
            src_scale = src_fp.abs().amax(dim=1).clamp(min=1e-12) / 127.0
            src_q = (src_fp / src_scale.unsqueeze(1)).round().clamp(-128, 127)
            acc = src_q @ w_int8.float().T
            result = acc * (src_scale.unsqueeze(1) * w_scales.unsqueeze(0))
            if bias is not None:
                result = result + bias.float()
            return result.to(torch_dtype).float()

        ref = dynamic_quant_matmul(x, w13, w13_scale, w13_bias)
        ref = self._apply_gated_activation(ref, activation)
        ref = dynamic_quant_matmul(ref, w2, w2_scale, w2_bias)

        output = ffn_output_buffer(x)
        torch.ops.zentorch.zentorch_fused_ffn_concat.out(
            output,
            x.clone(),
            w13,
            w2,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
            activation=activation,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
        )
        self.assertFalse(torch.isnan(output).any())
        self.assertEqual(
            output.float(), ref, **TOLERANCES["fused_bf16"]
        )

    # ------------------------------------------------------------------
    # DA8W4: packed s4 weights + per-group scales, bf16 activations.
    # ------------------------------------------------------------------
    @GroupMatmulTestCase.hypothesis_params_group_matmul_itr(
        dtype_list=["bfloat16"],
        **DA8W4_SHAPES,
    )
    @torch.inference_mode()
    def test_fused_ffn_da8w4(self, dtype, with_bias):
        quant = self.data.group_matmul_quant_moe_data
        x = quant["inputs"][0]
        w13 = quant["w13_packed"][0]
        w2 = quant["w2_packed"][0]
        w13_scale = quant["w13_scales"][0]
        w2_scale = quant["w2_scales"][0]
        w13_bias = quant["w13_bias"][0] if with_bias else None
        w2_bias = quant["w2_bias"][0] if with_bias else None
        activation = "silu"

        xq = dynamic_quant_dequant_per_token(x.float())
        gate_up = xq @ quant["dq_w13"][0].t()
        if w13_bias is not None:
            gate_up = gate_up + w13_bias.float()
        hidden = self._apply_gated_activation(gate_up, activation)
        hq = dynamic_quant_dequant_per_token(hidden)
        ref = hq @ quant["dq_w2"][0].t()
        if w2_bias is not None:
            ref = ref + w2_bias.float()

        output = ffn_output_buffer(x)
        torch.ops.zentorch.zentorch_fused_ffn_concat.out(
            output,
            x.clone(),
            w13,
            w2,
            w13_bias=w13_bias,
            w2_bias=w2_bias,
            activation=activation,
            w13_scale=w13_scale,
            w2_scale=w2_scale,
        )
        self.assertFalse(torch.isnan(output).any())
        self.assertEqual(
            output.float(), ref, **TOLERANCES["da8w4_fused"]
        )


if __name__ == "__main__":
    run_tests()
