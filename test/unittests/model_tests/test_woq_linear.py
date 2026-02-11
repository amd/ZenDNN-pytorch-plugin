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


def quantize_weight_per_channel(weight, qmin=-8, qmax=7):
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


def dequant_and_permute(weight_int8, zero_points_int8, scale_bf16):
    """Build the exact graph expected by WOQ pattern: (weight.to(bf16) - zp.to(bf16)) * scale then permute(1,0)."""
    w_bf16 = weight_int8.to(torch.bfloat16)
    zp_bf16 = zero_points_int8.to(torch.bfloat16)
    dq = (w_bf16 - zp_bf16) * scale_bf16
    return dq.permute(1, 0)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class WOQ_Linear_Model_NoBias(nn.Module):
    def __init__(self, out_features, in_features):
        super().__init__()
        original = torch.randn(out_features, in_features, dtype=torch.bfloat16)
        w_int8, scale_fp32, zp_int8 = quantize_weight_per_channel(original)
        self.register_buffer("weight", w_int8)
        self.register_buffer("zero_points", zp_int8)
        scale_bf16 = scale_fp32.to(torch.bfloat16)
        self.register_buffer("scale", scale_bf16)

    def forward(self, x):
        w_dequant = dequant_and_permute(self.weight, self.zero_points, self.scale)
        return torch.mm(x, w_dequant)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class WOQ_Linear_Model_With_Bias(nn.Module):

    def __init__(self, out_features, in_features):
        super().__init__()
        original = torch.randn(out_features, in_features, dtype=torch.bfloat16)
        w_int8, scale_fp32, zp_int8 = quantize_weight_per_channel(original)
        self.register_buffer("weight", w_int8)
        self.register_buffer("zero_points", zp_int8)
        scale_bf16 = scale_fp32.to(torch.bfloat16)
        self.register_buffer("scale", scale_bf16)
        self.register_buffer("bias", torch.randn(out_features, dtype=torch.bfloat16))

    def forward(self, x):
        w_dequant = dequant_and_permute(self.weight, self.zero_points, self.scale)
        return torch.addmm(self.bias, x, w_dequant)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQ_Linear_Model(Zentorch_TestCase):
    """Test that WOQ linear graph patterns (mm and addmm) are matched and replaced by zentorch_woq_linear."""

    @torch.inference_mode()
    def test_woq_linear_pattern_mm_no_bias(self):
        """Pattern 1: mm(input, permute(dequant(weight))) -> zentorch_woq_linear."""
        batch, in_features, out_features = 4, 64, 48
        model = WOQ_Linear_Model_NoBias(out_features, in_features).eval()
        x = torch.randn(batch, in_features, dtype=torch.bfloat16)

        eager_out = model(x)
        reset_dynamo()
        compiled = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_woq_linear"], 0)
        compiled_out = compiled(x)
        self.assertEqual(
            counters["zentorch"]["zentorch_woq_linear"],
            1,
            "WOQ mm pattern should be replaced by exactly one zentorch_woq_linear",
        )
        self.assertTrue(
            torch.allclose(compiled_out, eager_out, rtol=1e-2, atol=1e-2),
            "Compiled WOQ (mm) output should match eager.",
        )

    @torch.inference_mode()
    def test_woq_linear_pattern_addmm_with_bias(self):
        batch, in_features, out_features = 4, 64, 48
        model = WOQ_Linear_Model_With_Bias(out_features, in_features).eval()
        x = torch.randn(batch, in_features, dtype=torch.bfloat16)

        dequantized_output = model(x)
        reset_dynamo()
        compiled = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"]["zentorch_woq_linear"], 0)
        compiled_out = compiled(x)
        self.assertEqual(
            counters["zentorch"]["zentorch_woq_linear"],
            1,
            "WOQ addmm pattern should be replaced by exactly one zentorch_woq_linear",
        )
        self.assertEqual(dequantized_output, compiled_out, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
