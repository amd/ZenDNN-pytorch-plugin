# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: E402
    WOQTestCase,
    has_zentorch,
    zentorch,
    run_tests,
    freeze_opt,
    cpp_wrapper_opt,
    compare_inductor_vs_zentorch,
    WOQ_INT4_BATCH_RANGE,
    WOQ_INT4_IN_FEATURES_MULT_OPT,
    WOQ_INT4_OUT_FEATURES_OPT,
    WOQ_INT4_GROUP_SIZE_OPT,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_DA8W4(nn.Module):
    """Wraps zentorch_dynamic_qlinear (DA8W4 inferred from the packed int8
    [N, K/2] weight) so torch.compile places the op into the graph. Under
    backend='zentorch' + cpp_wrapper this exercises the
    aoti_torch_cpu_zentorch_dynamic_qlinear C-shim + ExternKernelAlloc lowering;
    under backend='inductor' it provides the reference compiled output."""

    def __init__(self, packed_weight, weight_scales, bias):
        super().__init__()
        self.register_buffer("packed_weight", packed_weight)
        self.register_buffer("weight_scales", weight_scales)
        # Register bias as a (possibly-None) buffer so it participates in
        # .to()/state_dict/freezing consistently with the other tensors.
        self.register_buffer("bias", bias)

    def forward(self, x):
        # DA8W4 is inferred from the packed int8 [N, K/2] weight (no selector).
        return torch.ops.zentorch.zentorch_dynamic_qlinear(
            x,
            self.packed_weight,
            self.weight_scales,
            self.bias,
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_DA8W4_DynamicQuantLinear_Model(WOQTestCase):
    """Compiles a DA8W4 zentorch_dynamic_qlinear model (mode inferred from the
    packed int8 weight) and
    checks that the backend='zentorch' output (including the cpp_wrapper
    AOTI-shim path) matches the backend='inductor' reference. A single
    Hypothesis test sweeps the per-group int4 shapes x bias x freeze x
    cpp_wrapper combinations."""

    def setUp(self):
        super().setUp()
        # ZenDNN caches the DA8W4 reordered weight by packed-weight buffer pointer;
        # keep each example's models alive so reused buffers can't cause stale hits.
        self._cmp_keepalive = []

    @staticmethod
    def _quantize_weight_per_group_s4(weight, group_size):
        """Symmetric per-group int4 quantization of a [N, K] weight -> packed s4.

        Returns (packed [N, K/2] int8, scales [G, N] f32). The s4 weight is
        packed exactly as the vLLM integration does: reuse the WOQ repack
        (8 int4 / int32) and reinterpret as int8.
        """
        n, k = weight.shape
        g = k // group_size
        w = weight.float().reshape(n, g, group_size)
        scale = (w.abs().amax(dim=2) / 8.0).clamp(min=1e-8)  # [N, G]
        qw = torch.clamp(torch.round(w / scale.unsqueeze(2)), -8, 7)
        w_s4 = qw.reshape(n, k).to(torch.int8)
        packed = torch.ops.zentorch.zentorch_woq_repack_weight(w_s4).view(
            torch.int8
        )  # [N, K/2]
        return packed, scale.t().contiguous()  # scales: [G, N]

    @WOQTestCase.hypothesis_params_woq_itr(
        # DA8W4 requires bf16 activations (f32 is rejected by the kernel).
        dtype_opt_list=["bfloat16"],
        batch_opt_list=WOQ_INT4_BATCH_RANGE,
        in_features_opt_list=WOQ_INT4_IN_FEATURES_MULT_OPT,
        out_features_opt_list=WOQ_INT4_OUT_FEATURES_OPT,
        group_size_opt_list=WOQ_INT4_GROUP_SIZE_OPT,
        bias_opt_list=[False, True],
        freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
        # A fresh cpp_wrapper compile far exceeds the default per-example
        # deadline; raise it so the deadline reflects compile cost.
        time_out=300000,
    )
    @torch.inference_mode()
    def test_da8w4_model(self, freeze_opt, cpp_wrapper):
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")

        weight = self.data.woq_weight  # [N, K] bf16
        group_size = self.data.group_size
        input_nd = self.data.woq_input  # [*, K] bf16
        bias = self.data.woq_bias  # [N] bf16 or None

        packed, weight_scales = self._quantize_weight_per_group_s4(weight, group_size)

        model = Custom_Model_DA8W4(packed, weight_scales, bias).eval()
        compare_inductor_vs_zentorch(self, model, (input_nd,), freeze_opt, cpp_wrapper)


if __name__ == "__main__":
    run_tests()
