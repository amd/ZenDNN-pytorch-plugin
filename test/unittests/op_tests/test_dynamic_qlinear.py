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

    def _per_token_reference(self, input_2d, weight_nk_f32, weight_scales):
        """Per-token: one scale per row of source. weight is [N, K]."""
        dq_weight = weight_nk_f32.t() * weight_scales.float()
        return torch.matmul(self._qdq_src(input_2d, dim=1), dq_weight)

    @torch.inference_mode()
    def test_per_token_fp32(self):
        """Per-token dynamic quantization, fp32 input."""
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")

        M, K, N = 16, 128, 64
        input = torch.randn(M, K, dtype=torch.float32)
        weight = torch.randint(-128, 128, (N, K), dtype=torch.int8)
        weight_scales = torch.rand(1, N, dtype=torch.float32) * 0.05

        ref = self._per_token_reference(
            input.reshape(M, K).float(), weight.float(), weight_scales
        ).to(input.dtype)
        out = torch.ops.zentorch.zentorch_dynamic_qlinear(
            input, weight, weight_scales, None,
        )

        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(ref, out, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    run_tests()
