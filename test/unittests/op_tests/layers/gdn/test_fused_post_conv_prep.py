# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for ``fused_post_conv_prep``."""

from __future__ import annotations

import sys
import unittest
from itertools import product
from pathlib import Path

import torch
from parameterized import parameterized

from test.unittests.op_tests.layers.gdn.helpers.shapes import (
    Qwen35_4B_GDN,
    common_seq_lens,
)

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from unittest_utils import (  # noqa: E402
    Zentorch_TestCase,
    default_tolerance,
    has_zentorch,
    run_tests,
)


DTYPES = [torch.float32, torch.bfloat16]
APPLY_L2NORM = [True, False]
OUTPUT_G_EXP = [False, True]
QWEN_L_VALUES = [64, 257]

_L2NORM_EPS = 1e-6
_SOFTPLUS_THRESHOLD = 20.0


def _softplus_manual(x: torch.Tensor, threshold: float) -> torch.Tensor:
    abs_x = x.abs()
    stable = x.clamp(min=0.0) + torch.log1p(torch.exp(-abs_x))
    return torch.where(x <= threshold, stable, x)


def fused_post_conv_prep_oracle(
    conv_output: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    apply_l2norm: bool = True,
    output_g_exp: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Independent oracle for ``fused_post_conv_prep``."""
    L, qkv_dim = conv_output.shape
    H = num_k_heads
    K = head_k_dim
    V = head_v_dim
    HV = A_log.shape[0]
    dtype = conv_output.dtype

    if qkv_dim != 2 * H * K + HV * V:
        raise ValueError(
            f"qkv_dim={qkv_dim} != 2*H*K + HV*V = {2 * H * K + HV * V}"
        )

    HK = H * K
    q_flat, k_flat, v_flat = torch.split(conv_output, [HK, HK, HV * V], dim=1)
    q = q_flat.reshape(L, H, K)
    k = k_flat.reshape(L, H, K)
    v = v_flat.reshape(L, HV, V)

    if apply_l2norm:
        q_f, k_f = q.float(), k.float()
        q_norm = torch.sqrt(q_f.pow(2).sum(dim=-1, keepdim=True) + _L2NORM_EPS)
        k_norm = torch.sqrt(k_f.pow(2).sum(dim=-1, keepdim=True) + _L2NORM_EPS)
        q = (q_f / q_norm).to(dtype)
        k = (k_f / k_norm).to(dtype)
    else:
        q = q.float().to(dtype)
        k = k.float().to(dtype)

    v = v.contiguous()
    q = q.contiguous()
    k = k.contiguous()

    a_f = a.float()
    b_f = b.float()
    A_log_f = A_log.float()
    dt_bias_f = dt_bias.float()

    sp = _softplus_manual(a_f + dt_bias_f, threshold=_SOFTPLUS_THRESHOLD)
    g = -torch.exp(A_log_f) * sp
    if output_g_exp:
        g = torch.exp(g)
    beta = torch.sigmoid(b_f)
    return q, k, v, g.contiguous(), beta.contiguous()


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _name_dtype_L_l2_g(method, _idx, params):
    dtype, L, apply_l2norm, output_g_exp = params.args
    l2_tag = "l2norm" if apply_l2norm else "no_l2norm"
    g_tag = "exp_g" if output_g_exp else "g"
    return f"{method.__name__}_{_dtype_name(dtype)}_L{L}_{l2_tag}_{g_tag}"


def _name_dtype_L(method, _idx, params):
    dtype, L = params.args
    return f"{method.__name__}_{_dtype_name(dtype)}_L{L}"


def _invoke_cpp(
    inputs: dict[str, torch.Tensor],
    *,
    num_k_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    apply_l2norm: bool = True,
    output_g_exp: bool = False,
) -> tuple[torch.Tensor, ...]:
    return torch.ops.zentorch.gdn_fused_post_conv_prep(
        inputs["conv_output"],
        inputs["a"],
        inputs["b"],
        inputs["A_log"],
        inputs["dt_bias"],
        num_k_heads,
        head_k_dim,
        head_v_dim,
        apply_l2norm,
        output_g_exp,
    )


def _make_inputs(
    L: int,
    *,
    H: int,
    HV: int,
    K: int,
    V: int,
    dtype: torch.dtype,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    qkv_dim = 2 * H * K + HV * V
    g = torch.Generator(device=device).manual_seed(L * 1000 + H + V)
    conv_output = torch.randn(L, qkv_dim, dtype=dtype, device=device, generator=g) * 0.5
    a = torch.randn(L, HV, dtype=dtype, device=device, generator=g)
    b = torch.randn(L, HV, dtype=dtype, device=device, generator=g)
    A_log = torch.randn(HV, dtype=torch.float32, device=device, generator=g) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device, generator=g) * 0.1
    if L > 0:
        a[0, 0] = 25.0
        if HV > 1:
            a[0, 1] = -25.0
    return {
        "conv_output": conv_output,
        "a": a,
        "b": b,
        "A_log": A_log,
        "dt_bias": dt_bias,
    }


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GDN_FusedPostConvPrep(Zentorch_TestCase):

    def _check_outputs(
        self,
        out_cpp: tuple[torch.Tensor, ...],
        out_oracle: tuple[torch.Tensor, ...],
        *,
        dtype: torch.dtype,
    ) -> None:
        q_c, k_c, v_c, g_c, beta_c = out_cpp
        q_o, k_o, v_o, g_o, beta_o = out_oracle
        self.assertEqual(q_c.dtype, dtype)
        self.assertEqual(q_o.dtype, dtype)
        self.assertEqual(k_c.dtype, dtype)
        self.assertEqual(k_o.dtype, dtype)
        self.assertEqual(v_c.dtype, dtype)
        self.assertEqual(v_o.dtype, dtype)
        self.assertEqual(g_c.dtype, torch.float32)
        self.assertEqual(g_o.dtype, torch.float32)
        self.assertEqual(beta_c.dtype, torch.float32)
        self.assertEqual(beta_o.dtype, torch.float32)

        for actual, expected, name in [
            (q_c, q_o, "q"), (k_c, k_o, "k"), (v_c, v_o, "v"),
            (g_c, g_o, "g"), (beta_c, beta_o, "beta"),
        ]:
            atol, rtol = default_tolerance(expected.dtype)
            self.assertEqual(
                actual, expected, atol=atol, rtol=rtol,
                msg=f"{name} mismatch (cpp vs oracle)",
            )

    @parameterized.expand(
        list(product(DTYPES, list(common_seq_lens()), APPLY_L2NORM, OUTPUT_G_EXP)),
        name_func=_name_dtype_L_l2_g,
    )
    def test_cpp_matches_oracle_small(
        self,
        dtype: torch.dtype,
        L: int,
        apply_l2norm: bool,
        output_g_exp: bool,
    ) -> None:
        cpu = torch.device("cpu")
        H, HV, K, V = 4, 8, 16, 32
        inputs = _make_inputs(L, H=H, HV=HV, K=K, V=V, dtype=dtype, device=cpu)

        out_cpp = _invoke_cpp(
            inputs,
            num_k_heads=H, head_k_dim=K, head_v_dim=V,
            apply_l2norm=apply_l2norm, output_g_exp=output_g_exp,
        )
        out_oracle = fused_post_conv_prep_oracle(
            **inputs,
            num_k_heads=H,
            head_k_dim=K,
            head_v_dim=V,
            apply_l2norm=apply_l2norm,
            output_g_exp=output_g_exp,
        )
        self._check_outputs(out_cpp, out_oracle, dtype=dtype)

    @parameterized.expand(
        list(product([torch.bfloat16], QWEN_L_VALUES)),
        name_func=_name_dtype_L,
    )
    def test_cpp_matches_oracle_qwen35_4b_shape(
        self,
        dtype: torch.dtype,
        L: int,
    ) -> None:
        s = Qwen35_4B_GDN
        cpu = torch.device("cpu")
        inputs = _make_inputs(
            L,
            H=s.num_k_heads,
            HV=s.num_v_heads,
            K=s.head_k_dim,
            V=s.head_v_dim,
            dtype=dtype,
            device=cpu,
        )
        out_cpp = _invoke_cpp(
            inputs,
            num_k_heads=s.num_k_heads,
            head_k_dim=s.head_k_dim,
            head_v_dim=s.head_v_dim,
        )
        out_oracle = fused_post_conv_prep_oracle(
            **inputs,
            num_k_heads=s.num_k_heads,
            head_k_dim=s.head_k_dim,
            head_v_dim=s.head_v_dim,
        )
        self._check_outputs(out_cpp, out_oracle, dtype=dtype)

    def test_softplus_threshold_branch(self) -> None:
        cpu = torch.device("cpu")
        L, HV = 4, 2
        a = torch.full((L, HV), 50.0, device=cpu)
        b = torch.zeros(L, HV, device=cpu)
        A_log = torch.zeros(HV, device=cpu)
        dt_bias = torch.zeros(HV, device=cpu)
        H, K, V = 1, 1, 1
        qkv_dim = 2 * H * K + HV * V
        conv_output = torch.zeros(L, qkv_dim, device=cpu)

        _, _, _, g, _ = torch.ops.zentorch.gdn_fused_post_conv_prep(
            conv_output, a, b, A_log, dt_bias,
            H, K, V, True, False,
        )
        expected = torch.full((L, HV), -50.0, device=cpu)
        atol, rtol = default_tolerance(expected.dtype)
        self.assertEqual(g, expected, atol=atol, rtol=rtol)


if __name__ == "__main__":
    run_tests()
