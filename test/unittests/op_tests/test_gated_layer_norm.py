# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
)


def gated_layer_norm(x, u, weight, bias, eps, silu_gate=False, concat=False):
    cols = x.shape[-1] * (3 if concat else 1)
    out = torch.empty(x.shape[0], cols, dtype=x.dtype)
    torch.ops.zentorch.zentorch_gated_layer_norm.out(
        x, u, weight, bias, eps, silu_gate=silu_gate, concat=concat, out=out
    )
    return out


def norm_mul(x, u, weight, bias, eps, silu_u, concat_ux):
    """`pytorch_norm_mul_dropout` of generative-recommenders at inference."""
    dtype = x.dtype
    u = u.float()
    if silu_u:
        u = F.silu(u)
    x = x.float()
    y = u * F.layer_norm(
        x,
        (x.shape[-1],),
        None if weight is None else weight.float(),
        None if bias is None else bias.float(),
        eps,
    )
    if concat_ux:
        y = torch.cat([u, x, y], dim=1)
    return y.to(dtype)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Gated_Layer_Norm(Zentorch_TestCase):
    def run_case(
        self,
        rows,
        dim,
        dtype=torch.bfloat16,
        silu=True,
        concat=True,
        affine=True,
        gate_view=False,
        gate_dtype=None,
        affine_dtype=None,
        weight=True,
        bias=True,
    ):
        torch.manual_seed(rows * 31 + dim)
        gate_dtype = gate_dtype or dtype
        affine_dtype = affine_dtype or dtype
        x = torch.randn(rows, dim).mul(2).add(0.5).to(dtype)
        if gate_view:
            # u as the first slice of a fused [u | v | q | k] GEMM output.
            u = torch.randn(rows, 4 * dim).mul(3).to(gate_dtype)[:, :dim]
        else:
            u = torch.randn(rows, dim).mul(3).to(gate_dtype)
        w = (
            torch.randn(dim).mul(0.5).add(1).to(affine_dtype)
            if affine and weight
            else None
        )
        b = torch.randn(dim).mul(0.5).to(affine_dtype) if affine and bias else None
        out = gated_layer_norm(x, u, w, b, 1e-6, silu_gate=silu, concat=concat)
        ref = norm_mul(x, u, w, b, 1e-6, silu, concat)
        self.assertEqual(out.shape, ref.shape)
        self.assertEqual(out.dtype, dtype)
        tol = {torch.bfloat16: 2e-2, torch.float16: 1e-2}.get(dtype, 1e-4)
        torch.testing.assert_close(out.float(), ref.float(), atol=tol, rtol=tol)
        if concat:
            # The x copy is exact.
            torch.testing.assert_close(out[:, dim : 2 * dim], x, atol=0, rtol=0)

    def test_dlrm_shape(self):
        self.run_case(300, 512, gate_view=True)

    def test_no_concat(self):
        self.run_case(65, 512, concat=False)

    def test_identity_gate(self):
        self.run_case(33, 128, silu=False)

    def test_no_affine(self):
        self.run_case(17, 96, affine=False)

    def test_tails(self):
        for dim in (1, 15, 17, 100):
            self.run_case(5, dim)

    def test_float32(self):
        self.run_case(40, 256, dtype=torch.float32, gate_view=True)

    def test_float16(self):
        self.run_case(40, 256, dtype=torch.float16, gate_view=True)

    def test_weight_or_bias_only(self):
        self.run_case(21, 96, bias=False)
        self.run_case(21, 96, weight=False)

    def test_broadcast_gate(self):
        x = torch.randn(9, 64, dtype=torch.bfloat16)
        u = torch.randn(1, 64, dtype=torch.bfloat16).expand(9, 64)
        out = gated_layer_norm(x, u, None, None, 1e-6, silu_gate=True, concat=True)
        torch.testing.assert_close(
            out.float(),
            norm_mul(x, u, None, None, 1e-6, True, True).float(),
            atol=2e-2,
            rtol=2e-2,
        )

    def test_single_and_empty_rows(self):
        self.run_case(1, 512, gate_view=True)
        x = torch.randn(0, 64, dtype=torch.bfloat16)
        out = gated_layer_norm(x, x, None, None, 1e-6, concat=True)
        self.assertEqual(out.shape, (0, 192))

    def test_strided_input(self):
        x = torch.randn(50, 256, dtype=torch.bfloat16)[:, :128]
        u = torch.randn(50, 128, dtype=torch.bfloat16)
        out = gated_layer_norm(x, u, None, None, 1e-6, silu_gate=True, concat=True)
        torch.testing.assert_close(
            out.float(),
            norm_mul(x, u, None, None, 1e-6, True, True).float(),
            atol=2e-2,
            rtol=2e-2,
        )

    def test_rejects_bad_inputs(self):
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        op = torch.ops.zentorch.zentorch_gated_layer_norm.out
        buf = torch.empty(4, 64, dtype=torch.bfloat16)
        cases = {
            "gate shape": (x, x[:, :32], None, None, 1e-6),
            "weight size": (x, x, torch.ones(32), None, 1e-6),
            "bias rank": (x, x, None, torch.ones(2, 32), 1e-6),
            "3-D input": (x.view(2, 2, 64), x.view(2, 2, 64), None, None, 1e-6),
            "int input": (x.int(), x, None, None, 1e-6),
            "eps 0": (x, x, None, None, 0.0),
            "eps below float": (x, x, None, None, 1e-50),
            "eps above float": (x, x, None, None, 1e40),
        }
        for name, args in cases.items():
            with self.subTest(name), self.assertRaises(RuntimeError):
                op(*args, out=buf)

    def test_compile(self):
        x = torch.randn(64, 128, dtype=torch.bfloat16)
        uvqk = torch.randn(64, 512, dtype=torch.bfloat16)

        def f(x, uvqk):
            u = uvqk[:, :128]
            return (
                gated_layer_norm(x, u, None, None, 1e-6, silu_gate=True, concat=True)
                * 2
            )

        out = torch.compile(f, backend="inductor", dynamic=True)(x, uvqk)
        torch.testing.assert_close(out, f(x, uvqk), atol=0, rtol=0)

    def test_compile_with_requires_grad(self):
        # A parameter that requires grad must not block inference compilation.
        # There is no backward for the out op, so differentiating the result
        # raises.
        w = torch.nn.Parameter(torch.ones(128))

        def f(x, uvqk):
            return gated_layer_norm(
                x, uvqk[:, :128], w, None, 1e-6, silu_gate=True, concat=True
            )

        cf = torch.compile(f, backend="inductor", dynamic=True)
        for rows in (64, 100, 257):
            x = torch.randn(rows, 128)
            uvqk = torch.randn(rows, 512)
            torch.testing.assert_close(cf(x, uvqk), f(x, uvqk), atol=1e-5, rtol=1e-5)
        with self.assertRaises(RuntimeError):
            cf(x, uvqk).sum().backward()

    def test_out_rejects_bad_buffer(self):
        x = torch.randn(4, 64, dtype=torch.bfloat16)
        op = torch.ops.zentorch.zentorch_gated_layer_norm.out
        cases = {
            "shape": torch.empty(4, 64, dtype=torch.bfloat16),
            "dtype": torch.empty(4, 192, dtype=torch.float32),
            "strided": torch.empty(4, 384, dtype=torch.bfloat16)[:, ::2],
        }
        for name, buf in cases.items():
            with self.subTest(name), self.assertRaises(RuntimeError):
                op(x, x, None, None, 1e-6, concat=True, out=buf)

    def test_no_backward(self):
        x = torch.randn(4, 64, requires_grad=True)
        out = gated_layer_norm(x, x.detach(), None, None, 1e-6)
        with self.assertRaises(RuntimeError):
            out.sum().backward()


if __name__ == "__main__":
    run_tests()
