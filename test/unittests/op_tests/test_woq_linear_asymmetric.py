# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    WOQTestCase,
    has_zentorch,
    zentorch,
    run_tests,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQLinear_Asymmetric(WOQTestCase):
    """Asymmetric WOQ op tests (per-group, opaque tensor path).

    Quantization follows the torchao tinygemm convention:
      scale = (max - min) / 15
      zero_point = min + scale * 8            (float-domain zero point)
      quantized   = round((val - min) / scale)  clamped to [0, 15]
      dequantized = (quantized - 8) * scale + zero_point

    Reference: torchao.quantization.quant_primitives._choose_qparams_affine_tinygemm
    and _quantize_affine_tinygemm.

    The test packs scale/zero_point into the ``(n_groups, N, 2)`` layout
    used by ``_weight_int4pack_mm_for_cpu`` (via pack_tinygemm_scales_and_zeros)
    and compares zentorch_woq_linear against native ``_weight_int4pack_mm_for_cpu``.
    """

    QUANT_MIN, QUANT_MAX = 0, 15
    MID_POINT = (QUANT_MAX + QUANT_MIN + 1) / 2.0  # 8.0

    def _quantize_weight(self, weight, group_size):
        """Per-group asymmetric quantization matching torchao tinygemm.

        Args:
            weight: (out_features, in_features) bfloat16 tensor.
            group_size: quantization group size.

        Returns:
            int_data: (out_features, in_features) int32, values in [0, 15].
            scale_and_zero: (n_groups, out_features, 2) bfloat16, tinygemm layout.
        """
        out_features, in_features = weight.shape
        n_groups = in_features // group_size

        w = weight.view(out_features, n_groups, group_size)
        min_val = w.amin(dim=2)
        max_val = w.amax(dim=2)

        scale = ((max_val - min_val) / float(self.QUANT_MAX - self.QUANT_MIN)).clamp(
            min=torch.finfo(weight.dtype).eps
        )
        zero_point = min_val + scale * self.MID_POINT
        scale = scale.to(weight.dtype)
        zero_point = zero_point.to(weight.dtype)

        min_val_recon = zero_point.unsqueeze(2) - scale.unsqueeze(2) * self.MID_POINT
        int_data = (
            torch.clamp(
                torch.round((w - min_val_recon) / scale.unsqueeze(2)),
                self.QUANT_MIN,
                self.QUANT_MAX,
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

    @WOQTestCase.hypothesis_params_woq_int4_itr()
    @torch.inference_mode()
    def test_woq_linear_asymmetric(
        self, tensor_seed, batch, in_features, out_features, group_size, with_bias
    ):
        """Asymmetric WOQ per-group, opaque tensor path."""
        unique_seed = (
            hash((tensor_seed, batch, in_features, out_features, group_size, with_bias))
            & 0x7FFFFFFF
        )
        g = torch.Generator().manual_seed(unique_seed)
        input = torch.randn(batch, in_features, dtype=torch.bfloat16, generator=g)
        weight = (
            torch.randn(out_features, in_features, dtype=torch.bfloat16, generator=g)
            * 0.02
        )
        bias = (
            torch.randn(out_features, dtype=torch.bfloat16, generator=g) * 0.01
            if with_bias
            else None
        )

        int_data, scale_and_zero = self._quantize_weight(weight, group_size)
        int4_packed = torch.ops.aten._convert_weight_to_int4pack_for_cpu(int_data, 1)

        pytorch_result = torch.ops.aten._weight_int4pack_mm_for_cpu(
            input, int4_packed, group_size, scale_and_zero
        )
        if bias is not None:
            pytorch_result = pytorch_result + bias

        repacked = torch.ops.zentorch.zentorch_woq_repack_from_int4pack(int4_packed)
        scale = scale_and_zero.select(2, 0).contiguous()
        zero_point = scale_and_zero.select(2, 1).contiguous()

        zentorch_result = torch.ops.zentorch.zentorch_woq_linear(
            input,
            repacked.t(),
            scale,
            zero_point,
            bias,
            zentorch_op_name="zentorch_woq_linear",
        )

        diff = (zentorch_result - pytorch_result).abs()
        tag = "with bias" if with_bias else "no bias"
        print(
            f"\n[{tag}] Max diff: {diff.max().item():.6f}, "
            f"Mean diff: {diff.mean().item():.6f}"
        )
        self.assertTrue(
            torch.allclose(zentorch_result, pytorch_result, rtol=0.1, atol=2.0),
            f"Asymmetric WOQ per-group ({tag}) does not match native int4 matmul. "
            f"Max diff: {diff.max().item():.6f}",
        )


if __name__ == "__main__":
    run_tests()
