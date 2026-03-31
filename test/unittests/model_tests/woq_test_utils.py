# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Shared utilities for WOQ linear and fusion tests."""

import unittest
import torch
from torch import nn

# Import has_zentorch for skipIf
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
try:
    from unittest_utils import has_zentorch
except ImportError:
    has_zentorch = False


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
    """
    out_features, in_features = weight.shape
    assert in_features % group_size == 0, "in_features must be divisible by group_size"
    n_groups = in_features // group_size

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
    """
    w_bf16 = weight_int8.to(torch.bfloat16)
    zp_bf16 = zero_points_int8.to(torch.bfloat16)
    dq = (w_bf16 - zp_bf16) * scale_bf16
    if view_shape is not None:
        dq = dq.view(view_shape)
    return dq.permute(1, 0)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class WOQ_Linear_Model(nn.Module):
    """WOQ model supporting per-channel and per-group quantization."""

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
        self.out_features = out_features

    def forward(self, x):
        w_dequant = dequant_and_permute(
            self.weight, self.zero_points, self.scale, self.view_shape
        )
        if self.bias is not None:
            return torch.addmm(self.bias, x, w_dequant)
        return torch.mm(x, w_dequant)
