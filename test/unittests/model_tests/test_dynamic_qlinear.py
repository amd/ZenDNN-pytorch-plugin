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
from unittest_utils import (  # noqa: 402
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
    freeze_opt,
    cpp_wrapper_opt,
    compare_inductor_vs_zentorch,
    counters,
    reset_dynamo,
    DYNAMIC_QLINEAR_K_OPT,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_DynamicQLinear(nn.Module):
    """Wraps zentorch_dynamic_qlinear so torch.compile places the op into the
    graph. Under backend='zentorch' + cpp_wrapper this exercises the
    aoti_torch_cpu_zentorch_dynamic_qlinear C-shim + ExternKernelAlloc lowering;
    under backend='inductor' it provides the reference compiled output."""

    def __init__(self, weight_int8, weight_scales, bias):
        super().__init__()
        self.register_buffer("weight_int8", weight_int8)
        self.register_buffer("weight_scales", weight_scales)
        # Register bias as a (possibly-None) buffer so it participates in
        # .to()/state_dict/freezing consistently with the other tensors.
        self.register_buffer("bias", bias)

    def forward(self, x):
        return torch.ops.zentorch.zentorch_dynamic_qlinear(
            x, self.weight_int8, self.weight_scales, self.bias,
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_DynamicQLinear_Model(QLinearTestCase):
    """Compiles a zentorch_dynamic_qlinear model and checks that the
    backend='zentorch' output (including the cpp_wrapper AOTI-shim path) matches
    the backend='inductor' reference. A single Hypothesis test sweeps the
    (dtype x bias x freeze x cpp_wrapper) combinations; a second test repeats it
    with non-contiguous weight / weight_scales / bias."""

    def _prepare_inputs(self, dtype, input_dim, q_weight_idx, bias_opt_idx):
        """Build (input, weight_int8, weight_scales, bias) from Hypothesis data,
        aligning K to a multiple of 4 (AOCL s8s8s32 kernel requirement). Returns
        None if the drawn shape can't yield a valid K."""
        input_nd = self.data.x_for_qlinear[dtype][input_dim]
        weight_int8 = self.data.y_int8[q_weight_idx]  # [N, K]
        k_orig = input_nd.shape[-1]
        k4 = (k_orig // 4) * 4
        if k4 < 4:
            return None
        input_nd = input_nd[..., :k4].contiguous()
        weight_int8 = weight_int8[:, :k4].contiguous()
        weight_scales = self.data.y_scales["per_channel"].unsqueeze(0)  # [1, N]
        bias = self.data.bias_for_qlinear[bias_opt_idx]
        if bias is not None:
            bias = bias.to(input_nd.dtype)
        return input_nd, weight_int8, weight_scales, bias

    @staticmethod
    def _make_noncontiguous(t):
        """Return a non-contiguous tensor with the same shape/values as `t` by
        materializing it into a 2x-wide buffer and taking a stride-2 view along
        the last dim (so stride[-1] == 2 != 1)."""
        if t is None:
            return None
        wide_shape = list(t.shape)
        wide_shape[-1] *= 2
        wide = torch.zeros(wide_shape, dtype=t.dtype)
        sl = [slice(None)] * t.dim()
        sl[-1] = slice(None, None, 2)
        wide[tuple(sl)] = t
        return wide[tuple(sl)]

    @QLinearTestCase.hypothesis_params_qlinear_itr(
        input_dim_opt_list=input_dim_opt,
        q_weight_list_opt_list=q_weight_list_opt,
        bias_opt_list=bias_opt,
        # The op always uses y_scales["per_channel"], so pin the granularity.
        q_granularity_opt_list=["per_channel"],
        q_zero_points_dtype_opt_list=q_zero_points_dtype_opt,
        q_linear_dtype_opt_list=q_linear_dtype_opt,
        dtype_list=qlinear_dtypes,
        freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
        # Constrain K so the drawn value is truncatable to a valid multiple of 4.
        kRange=Range(DYNAMIC_QLINEAR_K_OPT[0], DYNAMIC_QLINEAR_K_OPT[-1]),
        # A fresh cpp_wrapper compile far exceeds the default 10s per-example
        # deadline; raise it so the deadline reflects compile cost.
        time_out=300000,
    )
    @torch.inference_mode()
    def test_dynamic_qlinear_model(
        self, dtype, input_dim, q_weight_idx, bias_opt_idx, freeze_opt,
        cpp_wrapper,
    ):
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")
        # dynamic_qlinear input must be float32 or bfloat16.
        if dtype not in ("float32", "bfloat16"):
            self.skipTest(f"dynamic_qlinear input dtype {dtype} unsupported")

        prepared = self._prepare_inputs(
            dtype, input_dim, q_weight_idx, bias_opt_idx
        )
        if prepared is None:
            self.skipTest("K too small after aligning to a multiple of 4")
        input_nd, weight_int8, weight_scales, bias = prepared

        model = Custom_Model_DynamicQLinear(
            weight_int8, weight_scales, bias
        ).eval()
        compare_inductor_vs_zentorch(
            self, model, (input_nd,), freeze_opt, cpp_wrapper
        )

    @QLinearTestCase.hypothesis_params_qlinear_itr(
        input_dim_opt_list=input_dim_opt,
        q_weight_list_opt_list=q_weight_list_opt,
        bias_opt_list=bias_opt,
        q_granularity_opt_list=["per_channel"],
        q_zero_points_dtype_opt_list=q_zero_points_dtype_opt,
        q_linear_dtype_opt_list=q_linear_dtype_opt,
        dtype_list=qlinear_dtypes,
        freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
        kRange=Range(DYNAMIC_QLINEAR_K_OPT[0], DYNAMIC_QLINEAR_K_OPT[-1]),
        time_out=300000,
    )
    @torch.inference_mode()
    def test_dynamic_qlinear_noncontiguous_model(
        self, dtype, input_dim, q_weight_idx, bias_opt_idx, freeze_opt,
        cpp_wrapper,
    ):
        """Same backend comparison, but the weight / weight_scales / bias are
        non-contiguous. The kernel reads these via raw data_ptr() with hardcoded
        leading dims, so the lowering must pin contiguity -- both backends must
        still agree (and match the contiguous result)."""
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")
        if dtype not in ("float32", "bfloat16"):
            self.skipTest(f"dynamic_qlinear input dtype {dtype} unsupported")

        prepared = self._prepare_inputs(
            dtype, input_dim, q_weight_idx, bias_opt_idx
        )
        if prepared is None:
            self.skipTest("K too small after aligning to a multiple of 4")
        input_nd, weight_int8, weight_scales, bias = prepared

        weight_int8 = self._make_noncontiguous(weight_int8)
        weight_scales = self._make_noncontiguous(weight_scales)
        bias = self._make_noncontiguous(bias)
        self.assertFalse(weight_int8.is_contiguous())
        self.assertFalse(weight_scales.is_contiguous())
        if bias is not None:
            self.assertFalse(bias.is_contiguous())

        model = Custom_Model_DynamicQLinear(
            weight_int8, weight_scales, bias
        ).eval()
        compare_inductor_vs_zentorch(
            self, model, (input_nd,), freeze_opt, cpp_wrapper
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_DynamicQLinear_OutVariant_Lowering(QLinearTestCase):
    """Confirms the functional zentorch_dynamic_qlinear node lowers to the
    out-variant shim under torch.compile(backend='zentorch'). The lowering
    bumps counters['zentorch']['zentorch_dynamic_qlinear_out'] once per node,
    so a two-layer model must register exactly 2."""

    @staticmethod
    def _layer(N, K):
        weight = torch.randint(-8, 7, (N, K), dtype=torch.int8)
        scales = torch.rand(1, N, dtype=torch.float32) * 0.02 + 0.01
        bias = torch.rand(N, dtype=torch.float32)
        return weight, scales, bias

    @torch.inference_mode()
    def test_dynamic_qlinear_lowers_to_out_variant(self):
        if not zentorch._C.is_avx512_supported():
            self.skipTest("AVX512 not supported")

        w0, s0, b0 = self._layer(32, 16)
        w1, s1, b1 = self._layer(16, 32)

        class TwoLayer(nn.Module):
            def forward(self, x):
                h = torch.ops.zentorch.zentorch_dynamic_qlinear(x, w0, s0, b0)
                return torch.ops.zentorch.zentorch_dynamic_qlinear(h, w1, s1, b1)

        model = TwoLayer().eval()
        x = torch.randn(4, 16, dtype=torch.float32)

        ref = model(x)

        # Force a fresh compile so a FxGraphCache hit doesn't skip the lowering.
        from torch._inductor.utils import fresh_inductor_cache

        reset_dynamo()
        counters.clear()
        with fresh_inductor_cache():
            compiled = torch.compile(model, backend="zentorch")
            got = compiled(x)

        self.assertEqual(
            counters["zentorch"]["zentorch_dynamic_qlinear_out"], 2
        )
        self.assertTrue(torch.allclose(ref, got, atol=1e-2, rtol=1e-2))


if __name__ == "__main__":
    run_tests()
