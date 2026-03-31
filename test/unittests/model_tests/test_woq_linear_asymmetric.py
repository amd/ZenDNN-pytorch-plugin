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
    WOQTestCase,
)

QUANT_MIN, QUANT_MAX = 0, 15
MID_POINT = (QUANT_MAX + QUANT_MIN + 1) / 2.0  # 8.0


def quantize_weight_tinygemm(weight, group_size):
    """Per-group asymmetric quantization matching torchao tinygemm convention.

    Args:
        weight: (out_features, in_features) bfloat16.
        group_size: quantization group size.

    Returns:
        int_data: (out_features, in_features) int32, values in [0, 15].
        scale_and_zero: (n_groups, out_features, 2) bfloat16 (tinygemm layout).
    """
    out_features, in_features = weight.shape
    n_groups = in_features // group_size

    w = weight.view(out_features, n_groups, group_size)
    min_val = w.amin(dim=2)
    max_val = w.amax(dim=2)

    scale = ((max_val - min_val) / float(QUANT_MAX - QUANT_MIN)).clamp(
        min=torch.finfo(weight.dtype).eps
    )
    zero_point = min_val + scale * MID_POINT
    scale = scale.to(weight.dtype)
    zero_point = zero_point.to(weight.dtype)

    min_val_recon = zero_point.unsqueeze(2) - scale.unsqueeze(2) * MID_POINT
    int_data = (
        torch.clamp(
            torch.round((w - min_val_recon) / scale.unsqueeze(2)),
            QUANT_MIN,
            QUANT_MAX,
        )
        .to(torch.int32)
        .view(out_features, in_features)
    )

    scale_and_zero = (
        torch.cat([scale.unsqueeze(-1), zero_point.unsqueeze(-1)], dim=-1)
        .transpose(0, 1)
        .contiguous()
    )
    return int_data, scale_and_zero


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class WOQ_Linear_Asymmetric_Model(nn.Module):
    """Asymmetric WOQ model using Int4OpaqueTensor-style dispatch.

    The forward path calls aten._weight_int4pack_mm_for_cpu, which is the
    dispatch target when Int4OpaqueTensor is used with aten.linear.
    Weights are quantized using the torchao tinygemm convention.
    """

    def __init__(self, out_features, in_features, group_size, generator=None):
        super().__init__()
        original_weight = (
            torch.randn(
                out_features, in_features, dtype=torch.bfloat16, generator=generator
            )
            * 0.02
        )
        int_data, scale_and_zero = quantize_weight_tinygemm(original_weight, group_size)
        packed_weight = torch.ops.aten._convert_weight_to_int4pack_for_cpu(int_data, 1)

        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("scale_and_zero", scale_and_zero)
        self.group_size = group_size

    def forward(self, x):
        return torch._weight_int4pack_mm_for_cpu(
            x, self.packed_weight, self.group_size, self.scale_and_zero
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQ_Linear_Asymmetric(WOQTestCase):
    """Test that the Int4OpaqueTensor asymmetric WOQ pattern
    (aten._weight_int4pack_mm_for_cpu) is matched and replaced
    by zentorch_woq_linear."""

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
        diff = (compiled_out - eager_out).abs()
        print(
            f"\n[{pattern_description}] Max diff: {diff.max().item():.6f}, "
            f"Mean diff: {diff.mean().item():.6f}"
        )
        self.assertTrue(
            torch.allclose(compiled_out, eager_out, rtol=0.1, atol=2.0),
            f"Compiled {pattern_description} output should match eager. "
            f"Max diff: {diff.max().item():.6f}",
        )

    @WOQTestCase.hypothesis_params_woq_int4_itr(bias_opt_list=[False], time_out=30000)
    @torch.inference_mode()
    def test_woq_linear_asymmetric_int4_opaque(
        self, tensor_seed, batch, in_features, out_features, group_size
    ):
        """Asymmetric WOQ Int4OpaqueTensor pattern replacement."""
        unique_seed = (
            hash((tensor_seed, batch, in_features, out_features, group_size))
            & 0x7FFFFFFF
        )
        g = torch.Generator().manual_seed(unique_seed)
        model = WOQ_Linear_Asymmetric_Model(
            out_features, in_features, group_size, generator=g
        ).eval()
        x = torch.randn(batch, in_features, dtype=torch.bfloat16, generator=g)
        self._assert_woq_pattern_replaced(
            model,
            x,
            f"Asymmetric WOQ Int4OpaqueTensor "
            f"(N={out_features}, K={in_features}, gs={group_size})",
        )


if __name__ == "__main__":
    run_tests()
