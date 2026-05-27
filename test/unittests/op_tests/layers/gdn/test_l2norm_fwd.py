# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for ``l2norm_fwd``."""

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
SHAPES = [
    (1, 4),
    (16, 8),
    (32, 64),
    (1, 64, 16, 4),
    (2, 32, 4, 16),
    (5, 7, 13),
]
EPS_VALUES = [1e-6, 1e-3]
EPS_NAMES = {1e-6: "1e-6", 1e-3: "1e-3"}
SEQLENS_QWEN_T = [1, 64, 257]


def l2norm_fwd_oracle(
    x: torch.Tensor,
    eps: float = 1e-6,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Independent oracle for ``l2norm_fwd``."""
    if output_dtype is None:
        output_dtype = x.dtype
    x_f = x.float()
    norm = torch.linalg.vector_norm(x_f, ord=2, dim=-1, keepdim=True)
    denom = torch.sqrt(norm * norm + eps)
    return torch.div(x_f, denom).to(output_dtype)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _shape_name(shape: tuple[int, ...]) -> str:
    return "x".join(map(str, shape))


def _name_dtype_shape_eps(method, _idx, params):
    dtype, shape, eps = params.args
    return (
        f"{method.__name__}_{_dtype_name(dtype)}_{_shape_name(shape)}_"
        f"eps_{EPS_NAMES[eps]}"
    )


def _name_dtype_T(method, _idx, params):
    dtype, T = params.args
    return f"{method.__name__}_{_dtype_name(dtype)}_T{T}"


def _make_tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(*shape, dtype=dtype, device=device, generator=g) * 0.5


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GDN_L2NormFwd(Zentorch_TestCase):

    @parameterized.expand(
        list(product(DTYPES, SHAPES, EPS_VALUES)),
        name_func=_name_dtype_shape_eps,
    )
    def test_cpp_matches_oracle(
        self,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        eps: float,
    ) -> None:
        cpu = torch.device("cpu")
        x = _make_tensor(shape, dtype, cpu, seed=sum(shape))
        y_cpp = torch.ops.zentorch.gdn_l2norm_fwd(x, eps)
        y_oracle = l2norm_fwd_oracle(x, eps=eps)

        self.assertEqual(y_cpp.shape, x.shape)
        self.assertEqual(y_cpp.dtype, dtype)
        atol, rtol = default_tolerance(dtype)
        self.assertEqual(y_cpp, y_oracle, atol=atol, rtol=rtol)

    @parameterized.expand(
        [(torch.bfloat16, T) for T in SEQLENS_QWEN_T],
        name_func=_name_dtype_T,
    )
    def test_qwen35_4b_qk_shape(
        self,
        dtype: torch.dtype,
        T: int,
    ) -> None:
        s = Qwen35_4B_GDN
        cpu = torch.device("cpu")
        x = _make_tensor((1, T, s.num_k_heads, s.head_k_dim), dtype, cpu, seed=T)
        y_cpp = torch.ops.zentorch.gdn_l2norm_fwd(x, 1e-6)
        y_oracle = l2norm_fwd_oracle(x)
        atol, rtol = default_tolerance(dtype)
        self.assertEqual(y_cpp, y_oracle, atol=atol, rtol=rtol)

    def test_unit_norm_property(self) -> None:
        cpu = torch.device("cpu")
        x = _make_tensor((8, 32), torch.float32, cpu, seed=42)
        y = torch.ops.zentorch.gdn_l2norm_fwd(x, 1e-12)
        norms = torch.linalg.vector_norm(y, ord=2, dim=-1)
        self.assertEqual(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)

    def test_zero_input_zero_output(self) -> None:
        cpu = torch.device("cpu")
        x = torch.zeros(4, 8, dtype=torch.float32, device=cpu)
        y = torch.ops.zentorch.gdn_l2norm_fwd(x, 1e-6)
        self.assertTrue(torch.all(y == 0))

    def test_eps_inside_sqrt_kernel_form(self) -> None:
        cpu = torch.device("cpu")
        eps = 0.01
        x = torch.zeros(2, 4, dtype=torch.float32, device=cpu)
        x[0] = 1.0
        x[1] = 1e-3
        y_kernel_form = torch.ops.zentorch.gdn_l2norm_fwd(x, eps)

        sum_sq = x.pow(2).sum(dim=-1, keepdim=True)
        expected = x / torch.sqrt(sum_sq + eps)
        self.assertEqual(y_kernel_form, expected)

        expected_row1 = x[1] / torch.sqrt(torch.tensor(4e-6 + eps))
        self.assertEqual(y_kernel_form[1], expected_row1)

    def test_preserves_shape(self) -> None:
        cpu = torch.device("cpu")
        for shape in [(1, 4), (3, 1, 8), (2, 3, 4, 5)]:
            with self.subTest(shape=shape):
                x = _make_tensor(shape, torch.float32, cpu, seed=sum(shape))
                y = torch.ops.zentorch.gdn_l2norm_fwd(x, 1e-6)
                self.assertEqual(y.shape, x.shape)


if __name__ == "__main__":
    run_tests()
