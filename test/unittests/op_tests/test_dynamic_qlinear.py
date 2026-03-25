# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: E402
    Zentorch_TestCase,
    has_zentorch,
    zentorch,
    run_tests,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_DynamicQLinear(Zentorch_TestCase):
    """Test zentorch_dynamic_qlinear with per-token source, per-channel weight scales."""

    def _qdq_src(self, src, dim=None):
        """Quantize-dequantize src to S8: q = round(src/scale), dq = q*scale."""
        abs_max = src.abs().amax(dim=dim, keepdim=True).clamp(min=1e-12)
        scale = abs_max / 127.0
        return torch.clamp(torch.round(src / scale), -128, 127) * scale

    def _quantize_weight_per_channel(self, weight_float):
        """Quantize float weight [N, K] to per-channel symmetric qint8 via PyTorch."""
        scales = weight_float.abs().amax(dim=1).clamp(min=1e-12) / 127.0
        zero_points = torch.zeros(weight_float.size(0), dtype=torch.long)
        return torch.quantize_per_channel(
            weight_float, scales, zero_points, axis=0, dtype=torch.qint8
        )

    def _per_token_reference(self, input_2d, weight_q):
        """Per-token: one scale per row of source. weight_q is a per-channel quantized [N, K]."""
        dq_weight = weight_q.dequantize().t()
        return torch.matmul(self._qdq_src(input_2d, dim=1), dq_weight)

    @torch.inference_mode()
    def test_per_token_fp32(self):
        """Per-token dynamic quantization, fp32 input."""
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")

        M, K, N = 16, 128, 64
        input = torch.randn(M, K, dtype=torch.float32)
        weight_float = torch.randn(N, K, dtype=torch.float32)
        weight_q = self._quantize_weight_per_channel(weight_float)
        weight_int8 = weight_q.int_repr()
        weight_scales = weight_q.q_per_channel_scales().to(torch.float32).unsqueeze(0)

        ref = self._per_token_reference(
            input.reshape(M, K).float(), weight_q
        ).to(input.dtype)
        out = torch.ops.zentorch.zentorch_dynamic_qlinear(
            input, weight_int8, weight_scales, None,
        )

        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(ref, out, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    run_tests()
