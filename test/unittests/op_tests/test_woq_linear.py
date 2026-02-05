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
    has_zentorch,
    zentorch,
    run_tests,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQLinear(unittest.TestCase):
    qmin, qmax = -8, 7

    def quantize_weight(self, weight):
        """
        Quantize weight to int4 format with per-channel symmetric quantization (zero_point=0).

        Args:
            weight: Original weight tensor (out_features, in_features)

        Returns:
            quantized_weight: Quantized weight (int8)
            scale: Scale tensor for dequantization (out_features, 1) - float32
            zero_point: Zero point tensor for dequantization (out_features, 1) - float32 (zeros)
        """
        # Calculate per-channel max absolute value for symmetric quantization
        weight_absmax = weight.abs().max(dim=1, keepdim=True)[0]

        # Calculate scale for symmetric quantization (zero_point = 0)
        # Scale maps [-weight_absmax, weight_absmax] to [qmin, qmax]
        scale = weight_absmax / max(abs(self.qmin), abs(self.qmax))
        scale = torch.clamp(scale, min=1e-8).to(torch.float32)

        # Symmetric quantization: zero_point = 0 (must be int8)
        zero_point = torch.zeros_like(scale).to(torch.int8)

        # Quantize the weight
        quantized_weight = torch.clamp(
            torch.round(weight / scale), self.qmin, self.qmax
        ).to(torch.int8)

        return quantized_weight, scale, zero_point

    @torch.inference_mode()
    def test_woq_linear_accuracy(self):
        """Test accuracy of WOQ linear against PyTorch linear with dequantized weights."""
        input = torch.randn(24, 64, dtype=torch.bfloat16)

        # Create original weight
        original_weight = torch.randn(48, 64, dtype=torch.bfloat16)

        # Quantize weight
        quantized_weight, scale, zero_point = self.quantize_weight(original_weight)

        # Pack the quantized weight
        packed_weight = torch.ops.zentorch.zentorch_weight_from_int4pack_and_repack(
            quantized_weight
        )

        # Dequantize weight for reference (scale and zero_point are already (out_features, 1))
        dq_weight = (quantized_weight.to(torch.bfloat16) - zero_point) * scale

        # Run zentorch WOQ linear (expects scale and zero_point as (out_features, 1))
        zentorch_result = torch.ops.zentorch.zentorch_woq_linear(
            input, packed_weight, -1, scale, zero_point  # no bias
        )
        # Run PyTorch linear with dequantized weight
        pytorch_result = torch.nn.functional.linear(input, dq_weight.to(torch.bfloat16))

        # Compare results
        self.assertTrue(
            torch.allclose(zentorch_result, pytorch_result, rtol=1e-3, atol=1e-3),
            "Zentorch WOQ result does not match PyTorch DQ result.",
        )


if __name__ == "__main__":
    run_tests()
