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
    WOQTestCase,
    has_zentorch,
    zentorch,
    run_tests,
    WOQ_INT4_BATCH_RANGE,
    WOQ_INT4_IN_FEATURES_MULT_OPT,
    WOQ_INT4_OUT_FEATURES_OPT,
    WOQ_INT4_GROUP_SIZE_OPT,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_DA8W4_DynamicQuantLinear(WOQTestCase):
    """DA8W4 path of zentorch_dynamic_qlinear (inferred from packed int8 weight):

    dynamic per-token s8 activation x symmetric per-group s4 weight. Uses the
    same per-group int4 shape sweep as the WOQ int4 tests
    (group_size in {128, 256}, in_features in {256, 512, 1024}); both satisfy
    the kernel's (K/G) % 4 == 0 and the reused WOQ-repack's K % 8 == 0. The s4
    weight is packed exactly as the vLLM integration does -- by reusing
    zentorch_woq_repack_weight (8 int4 / int32) and reinterpreting the result as
    int8 [N, K/2] -- and validated against a dequant reference.
    """

    def setUp(self):
        super().setUp()
        # ZenDNN caches the DA8W4 reordered weight by packed-weight buffer pointer;
        # keep each example's packed weight alive so reused buffers can't cause stale hits.
        self._packed_keepalive = []

    def _qdq_src(self, src, dim=None):
        """Quantize-dequantize src to S8: q = round(src/scale), dq = q*scale."""
        abs_max = src.abs().amax(dim=dim, keepdim=True).clamp(min=1e-12)
        scale = abs_max / 127.0
        return torch.clamp(torch.round(src / scale), -128, 127) * scale

    @staticmethod
    def _quantize_weight_per_group_s4(weight, group_size):
        """Symmetric per-group int4 quantization of a [N, K] weight.

        Returns (w_s4 [N, K] int8 in [-8, 7], scales [G, N] f32,
        dq_weight [N, K] f32).
        """
        n, k = weight.shape
        g = k // group_size
        w = weight.float().reshape(n, g, group_size)
        # int4 symmetric: map [-absmax, absmax] onto [-8, 7] (divide by 8).
        absmax = w.abs().amax(dim=2)  # [N, G]
        scale = (absmax / 8.0).clamp(min=1e-8)  # [N, G]
        qw = torch.clamp(torch.round(w / scale.unsqueeze(2)), -8, 7)  # [N, G, gs]
        w_s4 = qw.reshape(n, k).to(torch.int8)
        dq_weight = (qw * scale.unsqueeze(2)).reshape(n, k)  # [N, K] f32
        scales = scale.t().contiguous()  # [G, N]
        return w_s4, scales, dq_weight

    @WOQTestCase.hypothesis_params_woq_itr(
        # DA8W4 requires bf16 activations (f32 is rejected by the kernel).
        dtype_opt_list=["bfloat16"],
        batch_opt_list=WOQ_INT4_BATCH_RANGE,
        in_features_opt_list=WOQ_INT4_IN_FEATURES_MULT_OPT,
        out_features_opt_list=WOQ_INT4_OUT_FEATURES_OPT,
        group_size_opt_list=WOQ_INT4_GROUP_SIZE_OPT,
        bias_opt_list=[False, True],
        # Op-level (eager) test -- no compile; pin the compile knobs.
        freeze_list=[False],
        cpp_wrapper_opt_list=[False],
    )
    @torch.inference_mode()
    def test_da8w4_per_group(self, freeze_opt, cpp_wrapper):
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")

        weight = self.data.woq_weight  # [N, K] bf16
        group_size = self.data.group_size
        input_nd = self.data.woq_input  # [*, K] bf16
        bias = self.data.woq_bias  # [N] bf16 or None

        w_s4, weight_scales, dq_weight = self._quantize_weight_per_group_s4(
            weight, group_size
        )
        n, k = w_s4.shape

        # Pack the s4 weight using the WOQ repack (8 int4 per int32) and
        # reinterpret the int32 [N, K/8] result as int8 -> [N, K/2].
        packed = torch.ops.zentorch.zentorch_woq_repack_weight(w_s4).view(
            torch.int8
        )  # [N, K/2]
        # Keep the const weight alive so its buffer isn't reused by a later
        # example (see setUp) -- avoids stale weight-cache hits.
        self._packed_keepalive.append(packed)

        # Reference: per-token qdq the activation, matmul with the dequantized
        # per-group weight, add bias.
        input_flat = input_nd.float().reshape(-1, k)
        ref = torch.matmul(self._qdq_src(input_flat, dim=1), dq_weight.t())
        if bias is not None:
            ref = ref + bias.float()
        ref = ref.reshape(input_nd.shape[:-1] + (n,))

        # Warm-up primes the const-weight reorder cache; verify the 2nd call.
        # DA8W4 is inferred from the packed int8 [N, K/2] weight (no selector).
        torch.ops.zentorch.zentorch_dynamic_qlinear(
            input_nd,
            packed,
            weight_scales,
            bias,
        )
        out = torch.ops.zentorch.zentorch_dynamic_qlinear(
            input_nd,
            packed,
            weight_scales,
            bias,
        )

        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(ref, out.float(), atol=2e-2, rtol=2e-2)


if __name__ == "__main__":
    run_tests()
