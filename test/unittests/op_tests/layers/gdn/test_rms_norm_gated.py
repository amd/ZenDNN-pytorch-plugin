# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for ``RMSNormGated`` (op: ``rms_norm_gated``)."""

from __future__ import annotations

import sys
import unittest
from itertools import product
from pathlib import Path

import torch
from parameterized import parameterized

from test.unittests.op_tests.layers.gdn.helpers.shapes import Qwen35_4B_GDN

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from unittest_utils import (  # noqa: E402
    Zentorch_TestCase,
    default_tolerance,
    has_zentorch,
    run_tests,
)


DTYPES = [torch.float32, torch.bfloat16]
M_VALUES = [1, 7, 64, 257]
N_VALUES = [32, 128]
ACTIVATIONS = ["silu", "swish", "sigmoid"]
QWEN_L_VALUES = [64, 257]


def _gate(z: torch.Tensor, activation: str) -> torch.Tensor:
    sig = torch.sigmoid(z)
    if activation in ("silu", "swish"):
        return z * sig
    if activation == "sigmoid":
        return sig
    raise ValueError(f"Unknown activation: {activation!r}")


def _rms_normalize_fp32(
    x_f: torch.Tensor,
    weight_f: torch.Tensor,
    eps: float,
    group_size: int | None,
) -> torch.Tensor:
    N = x_f.shape[-1]
    D = N if group_size is None else group_size
    if N % D != 0:
        raise ValueError(f"group_size={group_size} must divide last dim={N}")
    ngroups = N // D

    if ngroups == 1:
        sum_sq = (x_f * x_f).sum(dim=-1, keepdim=True)
        rstd = (sum_sq / D + eps).pow(-0.5)
        return x_f * rstd * weight_f
    leading_shape = x_f.shape[:-1]
    x_g = x_f.view(*leading_shape, ngroups, D)
    sum_sq = (x_g * x_g).sum(dim=-1, keepdim=True)
    rstd = (sum_sq / D + eps).pow(-0.5)
    return (x_g * rstd).view(*leading_shape, N) * weight_f


def rms_norm_gated_oracle(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    z: torch.Tensor | None = None,
    eps: float = 1e-6,
    group_size: int | None = None,
    norm_before_gate: bool = False,
    activation: str = "silu",
) -> torch.Tensor:
    """Independent oracle for ``RMSNormGated``."""
    if activation not in ("silu", "swish", "sigmoid"):
        raise ValueError(f"Unknown activation: {activation!r}")
    if z is not None and z.shape != x.shape:
        raise ValueError("z must have the same shape as x")

    out_dtype = x.dtype
    x_f = x.float()
    weight_f = weight.float()

    if z is None:
        return _rms_normalize_fp32(x_f, weight_f, eps, group_size).to(out_dtype)

    g = _gate(z.float(), activation)
    if not norm_before_gate:
        x_f = x_f * g
        return _rms_normalize_fp32(x_f, weight_f, eps, group_size).to(out_dtype)
    out_f = _rms_normalize_fp32(x_f, weight_f, eps, group_size)
    return (out_f * g).to(out_dtype)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _name_dtype_M_N_activation(method, _idx, params):
    dtype, M, N, activation = params.args
    return (
        f"{method.__name__}_{_dtype_name(dtype)}_"
        f"M{M}_N{N}_{activation}"
    )


def _name_dtype_L_activation(method, _idx, params):
    dtype, L, activation = params.args
    return f"{method.__name__}_{_dtype_name(dtype)}_L{L}_{activation}"


def _make_inputs(
    M: int,
    N: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
    with_z: bool = True,
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(M * 1000 + N)
    x = torch.randn(M, N, dtype=dtype, device=device, generator=g) * 0.5
    weight = torch.randn(N, dtype=dtype, device=device, generator=g) * 0.1 + 1.0
    z = (
        torch.randn(M, N, dtype=dtype, device=device, generator=g)
        if with_z else None
    )
    return {"x": x, "weight": weight, "z": z}


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GDN_RmsNormGated(Zentorch_TestCase):

    @parameterized.expand(
        list(product(DTYPES, M_VALUES, N_VALUES, ACTIVATIONS)),
        name_func=_name_dtype_M_N_activation,
    )
    def test_cpp_matches_oracle(
        self,
        dtype: torch.dtype,
        M: int,
        N: int,
        activation: str,
    ) -> None:
        cpu = torch.device("cpu")
        inputs = _make_inputs(M, N, dtype=dtype, device=cpu)

        out_cpp = torch.ops.zentorch.gdn_rms_norm_gated(
            inputs["x"], inputs["weight"], inputs["z"], 1e-6, activation,
        )
        out_oracle = rms_norm_gated_oracle(
            inputs["x"],
            inputs["weight"],
            z=inputs["z"],
            eps=1e-6,
            group_size=None,
            norm_before_gate=True,
            activation=activation,
        )

        self.assertEqual(out_cpp.shape, inputs["x"].shape)
        self.assertEqual(out_cpp.dtype, dtype)
        atol, rtol = default_tolerance(dtype)
        self.assertEqual(out_cpp, out_oracle, atol=atol, rtol=rtol)

    @parameterized.expand(
        list(product([torch.bfloat16], QWEN_L_VALUES, ACTIVATIONS)),
        name_func=_name_dtype_L_activation,
    )
    def test_qwen35_4b_call_site_shape(
        self,
        dtype: torch.dtype,
        L: int,
        activation: str,
    ) -> None:
        s = Qwen35_4B_GDN
        cpu = torch.device("cpu")
        M = L * s.num_v_heads
        N = s.head_v_dim
        inputs = _make_inputs(M, N, dtype=dtype, device=cpu)

        out_cpp = torch.ops.zentorch.gdn_rms_norm_gated(
            inputs["x"], inputs["weight"], inputs["z"],
            s.rms_norm_eps, activation,
        )
        out_oracle = rms_norm_gated_oracle(
            inputs["x"],
            inputs["weight"],
            z=inputs["z"],
            eps=s.rms_norm_eps,
            group_size=None,
            norm_before_gate=True,
            activation=activation,
        )
        atol, rtol = default_tolerance(dtype)
        self.assertEqual(out_cpp, out_oracle, atol=atol, rtol=rtol)

    def test_silu_equals_swish(self) -> None:
        cpu = torch.device("cpu")
        M, N = 16, 64
        inputs = _make_inputs(M, N, dtype=torch.float32, device=cpu)
        out_silu = torch.ops.zentorch.gdn_rms_norm_gated(
            inputs["x"], inputs["weight"], inputs["z"], 1e-6, "silu",
        )
        out_swish = torch.ops.zentorch.gdn_rms_norm_gated(
            inputs["x"], inputs["weight"], inputs["z"], 1e-6, "swish",
        )
        self.assertEqual(out_silu, out_swish, atol=0.0, rtol=0.0)

    def test_sigmoid_branch_uses_pure_sigmoid_not_silu(self) -> None:
        cpu = torch.device("cpu")
        M, N = 4, 8
        inputs = _make_inputs(M, N, dtype=torch.float32, device=cpu)
        out_sigmoid = torch.ops.zentorch.gdn_rms_norm_gated(
            inputs["x"], inputs["weight"], inputs["z"], 1e-6, "sigmoid",
        )
        out_silu = torch.ops.zentorch.gdn_rms_norm_gated(
            inputs["x"], inputs["weight"], inputs["z"], 1e-6, "silu",
        )
        diff = (out_sigmoid - out_silu).abs().max().item()
        self.assertGreater(diff, 1e-3)


if __name__ == "__main__":
    run_tests()
