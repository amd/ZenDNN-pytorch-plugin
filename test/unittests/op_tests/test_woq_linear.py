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
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQLinear(Zentorch_TestCase):
    qmin, qmax = -8, 7

    def quantize_weight(self, weight):
        """
        Quantize weight to int4 format with per-channel symmetric quantization.

        Args:
            weight: Original weight tensor (out_features, in_features)

        Returns:
            quantized_weight: Quantized weight (int8)
            scale: Scale tensor for dequantization (out_features, 1) - float32
            zero_point: Zero point tensor (out_features, 1) - int8, all zeros for symmetric
        """
        # Calculate per-channel max absolute value for symmetric quantization
        weight_absmax = weight.abs().max(dim=1, keepdim=True)[0]

        # Calculate scale for symmetric quantization.
        # Scale maps [-weight_absmax, weight_absmax] to [qmin, qmax]
        scale = weight_absmax / max(abs(self.qmin), abs(self.qmax))
        scale = torch.clamp(scale, min=1e-8).to(torch.float32)

        # Quantize the weight
        quantized_weight = torch.clamp(
            torch.round(weight / scale), self.qmin, self.qmax
        ).to(torch.int8)
        zero_point = torch.zeros_like(scale, dtype=torch.int8)
        return quantized_weight, scale, zero_point

    @torch.inference_mode()
    def test_woq_linear_accuracy(self):
        """Test accuracy of WOQ linear against PyTorch linear with dequantized weights."""
        input = torch.randn(24, 64, dtype=torch.bfloat16)

        # Create original weight
        original_weight = torch.randn(48, 64, dtype=torch.bfloat16)

        # Quantize weight
        quantized_weight, scale, _ = self.quantize_weight(original_weight)

        # Pack the quantized weight
        packed_weight = torch.ops.zentorch.zentorch_woq_repack_weight(
            quantized_weight
        )

        # Dequantize weight for reference (scale is already (out_features, 1))
        dq_weight = quantized_weight.to(torch.bfloat16) * scale

        # Run zentorch WOQ linear (expects scale as (out_features, 1))
        zentorch_result = torch.ops.zentorch.zentorch_woq_linear(
            input, packed_weight.transpose(0, 1), scale.transpose(0, 1), None, None  # no bias
        )
        # Run PyTorch linear with dequantized weight
        pytorch_result = torch.nn.functional.linear(input, dq_weight.to(torch.bfloat16))

        # Compare results
        self.assertTrue(
            torch.allclose(zentorch_result, pytorch_result, rtol=1e-3, atol=1e-3),
            "Zentorch WOQ result does not match PyTorch DQ result.",
        )

    def _woq_setup(self, M, K, N, bias=False):
        """Create input, packed weight, scales, and reference dq_weight."""
        input_t = torch.randn(M, K, dtype=torch.bfloat16)
        original_weight = torch.randn(N, K, dtype=torch.bfloat16)
        quantized_weight, scale, _ = self.quantize_weight(original_weight)
        packed_weight = torch.ops.zentorch.zentorch_woq_repack_weight(
            quantized_weight
        )
        packed_weight_t = packed_weight.transpose(0, 1)
        scale_t = scale.transpose(0, 1).contiguous()
        dq_weight = quantized_weight.to(torch.bfloat16) * scale.to(torch.bfloat16)
        bias_t = (
            torch.randn(N, dtype=torch.bfloat16) if bias else None
        )
        return input_t, packed_weight_t, scale_t, dq_weight, bias_t

    def _assert_woq_fused_output_sanity(self, result, M, N, op_name):
        """Assert fused WOQ op output has correct shape, dtype, and finite values."""
        self.assertEqual(result.shape, (M, N), f"{op_name}: wrong output shape")
        self.assertEqual(result.dtype, torch.bfloat16, f"{op_name}: wrong output dtype")
        self.assertTrue(torch.isfinite(result).all(), f"{op_name}: output contains inf/nan")

    # Fused post-ops (GELU, binary) can have slightly larger numerical error than
    # plain WOQ linear; use rtol/atol 1e-2 for comparison with dq_output reference.
    _fused_rtol, _fused_atol = 1e-2, 1e-2

    @torch.inference_mode()
    def test_woq_linear_gelu_tanh_accuracy(self):
        """Test accuracy of WOQ linear + GELU(tanh) against dq linear + gelu(tanh)."""
        M, K, N = 24, 64, 48
        input_t, packed_weight_t, scale_t, dq_weight, bias_t = self._woq_setup(
            M, K, N, bias=False
        )
        zentorch_result = torch.ops.zentorch.zentorch_woq_linear_gelu_tanh(
            input_t, packed_weight_t, scale_t, None, None
        )
        dq_output = torch.nn.functional.linear(input_t, dq_weight.to(torch.bfloat16))
        pytorch_result = torch.nn.functional.gelu(dq_output, approximate="tanh")
        self._assert_woq_fused_output_sanity(zentorch_result, M, N, "zentorch_woq_linear_gelu_tanh")
        self.assertTrue(
            torch.allclose(zentorch_result, pytorch_result, rtol=self._fused_rtol, atol=self._fused_atol),
            "zentorch_woq_linear_gelu_tanh does not match dq linear + gelu(tanh).",
        )

    @torch.inference_mode()
    def test_woq_linear_gelu_erf_accuracy(self):
        """Test accuracy of WOQ linear + GELU(erf) against dq linear + gelu(none)."""
        M, K, N = 24, 64, 48
        input_t, packed_weight_t, scale_t, dq_weight, bias_t = self._woq_setup(
            M, K, N, bias=False
        )
        zentorch_result = torch.ops.zentorch.zentorch_woq_linear_gelu_erf(
            input_t, packed_weight_t, scale_t, None, None
        )
        dq_output = torch.nn.functional.linear(input_t, dq_weight.to(torch.bfloat16))
        pytorch_result = torch.nn.functional.gelu(dq_output, approximate="none")
        self._assert_woq_fused_output_sanity(zentorch_result, M, N, "zentorch_woq_linear_gelu_erf")
        self.assertTrue(
            torch.allclose(zentorch_result, pytorch_result, rtol=self._fused_rtol, atol=self._fused_atol),
            "zentorch_woq_linear_gelu_erf does not match dq linear + gelu(erf).",
        )

    @torch.inference_mode()
    def test_woq_linear_mul_add_accuracy(self):
        """Test accuracy of WOQ linear + mul + add against dq linear then (out * mul) + add."""
        M, K, N = 24, 64, 48
        input_t, packed_weight_t, scale_t, dq_weight, _ = self._woq_setup(
            M, K, N, bias=False
        )
        mul_input = torch.randn(M, N, dtype=torch.bfloat16)
        add_input = torch.randn(M, N, dtype=torch.bfloat16)
        zentorch_result = torch.ops.zentorch.zentorch_woq_linear_mul_add(
            input_t, packed_weight_t, scale_t, None, mul_input, add_input, None
        )
        dq_output = torch.nn.functional.linear(input_t, dq_weight.to(torch.bfloat16))
        pytorch_result = dq_output * mul_input + add_input
        self._assert_woq_fused_output_sanity(zentorch_result, M, N, "zentorch_woq_linear_mul_add")
        self.assertTrue(
            torch.allclose(zentorch_result, pytorch_result, rtol=self._fused_rtol, atol=self._fused_atol),
            "zentorch_woq_linear_mul_add does not match dq linear then (out * mul) + add.",
        )

    @torch.inference_mode()
    def test_woq_linear_add_add_accuracy(self):
        """Test accuracy of WOQ linear + add + add against dq linear then out + add + add_2."""
        M, K, N = 24, 64, 48
        input_t, packed_weight_t, scale_t, dq_weight, _ = self._woq_setup(
            M, K, N, bias=False
        )
        add_input = torch.randn(M, N, dtype=torch.bfloat16)
        add_input_2 = torch.randn(M, N, dtype=torch.bfloat16)
        zentorch_result = torch.ops.zentorch.zentorch_woq_linear_add_add(
            input_t, packed_weight_t, scale_t, None, add_input, add_input_2, None
        )
        dq_output = torch.nn.functional.linear(input_t, dq_weight.to(torch.bfloat16))
        pytorch_result = dq_output + add_input + add_input_2
        self._assert_woq_fused_output_sanity(zentorch_result, M, N, "zentorch_woq_linear_add_add")
        self.assertTrue(
            torch.allclose(zentorch_result, pytorch_result, rtol=self._fused_rtol, atol=self._fused_atol),
            "zentorch_woq_linear_add_add does not match dq linear then out + add + add_2.",
        )


if __name__ == "__main__":
    run_tests()
