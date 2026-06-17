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
    QLinearTestCase,
    Range,
    has_zentorch,
    zentorch,
    run_tests,
    qlinear_dtypes,
    input_dim_opt,
    q_weight_list_opt,
    bias_opt,
    q_zero_points_dtype_opt,
    q_linear_dtype_opt,
    DYNAMIC_QLINEAR_K_OPT,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_DynamicQLinear(QLinearTestCase):
    """Test zentorch_dynamic_qlinear with per-token source, per-channel weight scales."""

    def _qdq_src(self, src, dim=None):
        """Quantize-dequantize src to S8: q = round(src/scale), dq = q*scale."""
        abs_max = src.abs().amax(dim=dim, keepdim=True).clamp(min=1e-12)
        scale = abs_max / 127.0
        return torch.clamp(torch.round(src / scale), -128, 127) * scale

    @QLinearTestCase.hypothesis_params_qlinear_itr(
        input_dim_opt_list=input_dim_opt,
        q_weight_list_opt_list=q_weight_list_opt,
        bias_opt_list=bias_opt,
        # This test always uses y_scales["per_channel"], so constrain the drawn
        # granularity to per_channel to match the operator contract being validated.
        q_granularity_opt_list=["per_channel"],
        q_zero_points_dtype_opt_list=q_zero_points_dtype_opt,
        q_linear_dtype_opt_list=q_linear_dtype_opt,
        dtype_list=qlinear_dtypes,
        # Constrain K to [DYNAMIC_QLINEAR_K_OPT[0], DYNAMIC_QLINEAR_K_OPT[-1]] so
        # that the drawn K is always truncatable to a valid multiple of 4.
        kRange=Range(DYNAMIC_QLINEAR_K_OPT[0], DYNAMIC_QLINEAR_K_OPT[-1]),
    )
    @torch.inference_mode()
    def test_per_token_fp32(self, input_dim, q_weight_idx, bias_opt_idx, **kwargs):
        """Per-token dynamic quantization, fp32 input, using hypothesis-generated data.

        K is truncated to the nearest lower multiple of 4 because the underlying
        AOCL s8s8s32os32 kernel requires K % 4 == 0.
        """
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")

        input_2d = self.data.x_for_qlinear["float32"][input_dim]
        # y_int8[0]: shape [N, K], non-contiguous (created as [K, N] then .t())
        # y_int8[1]: shape [N, K], contiguous
        # Both variants are normalized by .contiguous() below, so either is safe.
        weight_int8 = self.data.y_int8[q_weight_idx]  # [N, K]

        # Align K to the nearest lower multiple of 4 (kernel requirement)
        k_orig = input_2d.shape[-1]
        k4 = (k_orig // 4) * 4
        if k4 < 4:
            self.skipTest(
                f"K={k_orig} rounds down to {k4}, which is too small; skipping"
            )
        # Trim input and weight to k4; .contiguous() ensures correct memory layout.
        input_2d = input_2d[..., :k4].contiguous()
        weight_int8 = weight_int8[:, :k4].contiguous()

        # per-channel scales shape [N]; dynamic_qlinear expects [1, N]
        weight_scales = self.data.y_scales["per_channel"].unsqueeze(0)  # [1, N]
        bias = self.data.bias_for_qlinear[bias_opt_idx]

        # Reference: dequantize weight, apply per-token qdq to input, then add bias.
        dq_weight = weight_int8.float() * self.data.y_scales["per_channel"].unsqueeze(1)
        input_flat = input_2d.float().reshape(-1, k4)
        ref = torch.matmul(self._qdq_src(input_flat, dim=1), dq_weight.t())
        if bias is not None:
            ref = ref + bias
        # Reshape reference from [B*M, N] back to the expected output shape
        # (..., N), preserving the input's batch/sequence dimensions.
        out_shape = input_2d.shape[:-1] + (weight_int8.shape[0],)
        ref = ref.reshape(out_shape)

        # Warm-up: first call primes weight packing; only the second call's output is verified.
        torch.ops.zentorch.zentorch_dynamic_qlinear(
            input_2d, weight_int8, weight_scales, bias,
        )
        out = torch.ops.zentorch.zentorch_dynamic_qlinear(
            input_2d, weight_int8, weight_scales, bias,
        )

        self.assertEqual(out.dtype, torch.float32)
        torch.testing.assert_close(ref, out, atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    run_tests()
