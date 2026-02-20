# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    has_zentorch,
    reset_dynamo,
    run_tests,
    counters,
    Zentorch_TestCase,
)


def symmetric_quantize_weight_per_channel(weight, qmin=-8, qmax=7):
    """
    Quantize weight to int4 format with per-channel symmetric quantization.
    Returns quantized_weight (int8), scale (float32), zero_point (int8).
    Weight shape: (out_features, in_features).
    """
    weight_absmax = weight.abs().max(dim=1, keepdim=True)[0]
    scale = weight_absmax / max(abs(qmin), abs(qmax))
    scale = torch.clamp(scale, min=1e-8).to(torch.float32)
    zero_point = torch.zeros_like(scale).to(torch.int8)
    quantized_weight = torch.clamp(
        torch.round(weight / scale), qmin, qmax
    ).to(torch.int8)
    return quantized_weight, scale, zero_point


def symmetric_quantize_weight_per_group(weight, group_size, qmin=-8, qmax=7):
    """
    Quantize weight to int4 format with per-group symmetric quantization.
    Returns quantized_weight (int8), scale (float32), zero_point (int8).
    Weight shape: (out_features, in_features).
    The weight is reshaped to (out_features, n_groups, group_size) for group-wise
    quantization, and the returned tensors are 3D.
    """
    out_features, in_features = weight.shape
    assert in_features % group_size == 0, "in_features must be divisible by group_size"
    n_groups = in_features // group_size

    # Reshape to (out_features, n_groups, group_size)
    weight_grouped = weight.view(out_features, n_groups, group_size)
    weight_absmax = weight_grouped.abs().max(dim=2, keepdim=True)[0]
    scale = weight_absmax / max(abs(qmin), abs(qmax))
    scale = torch.clamp(scale, min=1e-8).to(torch.float32)
    zero_point = torch.zeros(out_features, n_groups, 1, dtype=torch.int8)
    quantized_weight = torch.clamp(
        torch.round(weight_grouped / scale), qmin, qmax
    ).to(torch.int8)

    return quantized_weight, scale, zero_point


def dequant_and_permute(weight_int8, zero_points_int8, scale_bf16, view_shape=None):
    """
    Dequantize weights to bf16, optionally reshape, and permute.

    Per-channel path (view_shape=None):
        (weight.to(bf16) - zp.to(bf16)) * scale -> permute(1, 0)
    Per-group path (view_shape provided):
        (weight.to(bf16) - zp.to(bf16)) * scale -> view(out, in) -> permute(1, 0)
    """
    w_bf16 = weight_int8.to(torch.bfloat16)
    zp_bf16 = zero_points_int8.to(torch.bfloat16)
    dq = (w_bf16 - zp_bf16) * scale_bf16
    if view_shape is not None:
        dq = dq.view(view_shape)
    return dq.permute(1, 0)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class WOQ_Linear_Model(nn.Module):
    """WOQ model supporting per-channel and per-group quantization,
    Args:
        out_features: Number of output features.
        in_features:  Number of input features.
        group_size:   If *None*, use per-channel quantization (2-D weights).
                      Otherwise, use per-group quantization with the given
                      group size (3-D weights reshaped via *view*).
        bias:         Whether to include an additive bias (addmm vs mm).
    """

    def __init__(self, out_features, in_features, group_size=None, bias=False):
        super().__init__()
        original = torch.randn(out_features, in_features, dtype=torch.bfloat16)
        if group_size is not None:
            w_int8, scale_fp32, zp_int8 = symmetric_quantize_weight_per_group(
                original, group_size
            )
            self.view_shape = (out_features, in_features)
        else:
            w_int8, scale_fp32, zp_int8 = symmetric_quantize_weight_per_channel(original)
            self.view_shape = None

        self.register_buffer("weight", w_int8)
        self.register_buffer("zero_points", zp_int8)
        self.register_buffer("scale", scale_fp32.to(torch.bfloat16))
        self.register_buffer(
            "bias",
            torch.randn(out_features, dtype=torch.bfloat16) if bias else None,
        )

    def forward(self, x):
        w_dequant = dequant_and_permute(
            self.weight, self.zero_points, self.scale, self.view_shape
        )
        if self.bias is not None:
            return torch.addmm(self.bias, x, w_dequant)
        return torch.mm(x, w_dequant)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQ_Linear(Zentorch_TestCase):
    """Test that WOQ linear graph patterns (per-channel and per-group,
    mm and addmm) are matched and replaced by zentorch_woq_linear."""

    def _assert_woq_pattern_replaced(self, model, x, pattern_description):
        eager_out = model(x)
        reset_dynamo()
        compiled = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_woq_linear"], 0)
        compiled_out = compiled(x)
        self.assertEqual(
            counters["zentorch"]["zentorch_woq_linear"],
            1,
            f"{pattern_description} should be replaced by exactly one "
            "zentorch_woq_linear",
        )
        self.assertTrue(
            torch.allclose(compiled_out, eager_out, rtol=1e-2, atol=1e-2),
            f"Compiled {pattern_description} output should match eager.",
        )

    @torch.inference_mode()
    def test_woq_linear_per_channel_mm_no_bias(self):
        batch, in_features, out_features = 4, 64, 48
        model = WOQ_Linear_Model(out_features, in_features, group_size=None, bias=False).eval()
        x = torch.randn(batch, in_features, dtype=torch.bfloat16)
        self._assert_woq_pattern_replaced(model, x, "WOQ per-channel mm")

    @torch.inference_mode()
    def test_woq_linear_per_channel_addmm_with_bias(self):
        batch, in_features, out_features = 4, 64, 48
        model = WOQ_Linear_Model(out_features, in_features, group_size=None, bias=True).eval()
        x = torch.randn(batch, in_features, dtype=torch.bfloat16)
        self._assert_woq_pattern_replaced(model, x, "WOQ per-channel addmm")

    @torch.inference_mode()
    def test_woq_linear_per_group_mm_no_bias(self):
        batch, in_features, out_features, group_size = 4, 64, 48, 16
        model = WOQ_Linear_Model(
            out_features, in_features, group_size=group_size, bias=False
        ).eval()
        x = torch.randn(batch, in_features, dtype=torch.bfloat16)
        self._assert_woq_pattern_replaced(model, x, "WOQ per-group mm")

    @torch.inference_mode()
    def test_woq_linear_per_group_addmm_with_bias(self):
        batch, in_features, out_features, group_size = 4, 64, 48, 16
        model = WOQ_Linear_Model(
            out_features, in_features, group_size=group_size, bias=True
        ).eval()
        x = torch.randn(batch, in_features, dtype=torch.bfloat16)
        self._assert_woq_pattern_replaced(model, x, "WOQ per-group addmm")


if __name__ == "__main__":
    run_tests()
