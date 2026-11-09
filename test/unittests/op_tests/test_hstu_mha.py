# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import math
import os
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


def valid_attn_mask(
    n, num_targets, max_attn_len, contextual_seq_len, causal, min_full_attn_seq_len=0
):
    """`_get_valid_attn_mask` of generative-recommenders for one sequence."""
    ids = torch.arange(n).view(1, n)
    max_ids = torch.tensor(n).view(1, 1)
    if contextual_seq_len > 0:
        ids = torch.clamp(ids - contextual_seq_len + 1, min=0)
        max_ids = max_ids - contextual_seq_len + 1
    if num_targets is not None:
        max_ids = max_ids - num_targets
        ids = torch.clamp(ids, max=max_ids)
    row_ids = ids.view(n, 1).expand(n, n)
    col_ids = ids.view(1, n).expand(n, n)
    dist = row_ids - col_ids
    if not causal:
        dist = dist.abs()
    mask = torch.eye(n, dtype=torch.bool) | (dist > 0)
    if max_attn_len > 0:
        if min_full_attn_seq_len > 0:
            mask = mask & (
                (dist <= max_attn_len) | (row_ids >= max_ids - min_full_attn_seq_len)
            )
        else:
            mask = mask & (dist <= max_attn_len)
    if contextual_seq_len > 0:
        mask = mask | ((row_ids == 0) & (col_ids < max_ids))
    return mask


def reference(
    max_seq_len,
    alpha,
    q,
    k,
    v,
    offsets,
    causal,
    num_targets,
    attn_scale,
    max_attn_len,
    contextual_seq_len,
    delta=0,
    min_full_attn_seq_len=0,
):
    outs = []
    for b in range(offsets.numel() - 1):
        s, e = int(offsets[b]), int(offsets[b + 1])
        n = e - s
        t = None if num_targets is None else int(num_targets[b])
        mask = valid_attn_mask(
            n, t, max_attn_len, contextual_seq_len, causal, min_full_attn_seq_len
        )
        if delta:
            qb = q[b * delta : (b + 1) * delta].double()
            mask = mask[n - delta :]
        else:
            qb = q[s:e].double()
        a = torch.einsum("xha,yha->hxy", qb, k[s:e].double()) * alpha
        if attn_scale is None:
            a = F.silu(a) / max_seq_len
        elif attn_scale.dim() == 0:
            a = F.silu(a) * attn_scale.double()
        else:
            a = F.silu(a) * attn_scale[s:e].double().view(1, -1, 1)
        a = a * mask
        outs.append(torch.einsum("hxy,yhv->xhv", a, v[s:e].double()))
    return torch.cat(outs)


def hstu_mha(
    max_seq_len,
    alpha,
    q,
    k,
    v,
    seq_offsets,
    causal=True,
    num_targets=None,
    attn_scale=None,
    max_attn_len=0,
    contextual_seq_len=0,
    min_full_attn_seq_len=0,
    out=None,
):
    if out is None:
        out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], dtype=q.dtype)
    torch.ops.zentorch.zentorch_hstu_mha.out(
        max_seq_len,
        alpha,
        q,
        k,
        v,
        seq_offsets,
        causal,
        num_targets,
        attn_scale,
        max_attn_len,
        contextual_seq_len,
        min_full_attn_seq_len,
        out=out,
    )
    return out


def delta_hstu_mha(
    max_seq_len,
    alpha,
    delta_q,
    k,
    v,
    seq_offsets,
    num_targets=None,
    max_attn_len=0,
    contextual_seq_len=0,
    out=None,
):
    if out is None:
        out = torch.empty(
            delta_q.shape[0], k.shape[1], v.shape[-1], dtype=delta_q.dtype
        )
    torch.ops.zentorch.zentorch_delta_hstu_mha.out(
        max_seq_len,
        alpha,
        delta_q,
        k,
        v,
        seq_offsets,
        num_targets,
        max_attn_len,
        contextual_seq_len,
        out=out,
    )
    return out


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
@unittest.skipIf(
    bool(os.environ.get("ZENDNNL_HSTU_ATTENTION_KERNEL")),
    "ZENDNNL_HSTU_ATTENTION_KERNEL overrides the kernel",
)
class Test_HSTU_MHA(Zentorch_TestCase):
    H, D, V = 2, 64, 32

    def make(self, lens, dtype=torch.bfloat16, fused=False):
        offsets = torch.zeros(len(lens) + 1, dtype=torch.long)
        offsets[1:] = torch.cumsum(torch.tensor(lens), 0)
        L = int(offsets[-1])
        H, D, V = self.H, self.D, self.V
        if fused:
            uvqk = torch.randn(L, H * (2 * V + 2 * D)).to(dtype)
            _, v, q, k = torch.split(uvqk, [H * V, H * V, H * D, H * D], dim=1)
            return q.view(L, H, D), k.view(L, H, D), v.view(L, H, V), offsets
        q = torch.randn(L, H, D).to(dtype)
        k = torch.randn(L, H, D).to(dtype)
        v = torch.randn(L, H, V).to(dtype)
        return q, k, v, offsets

    def check(self, got, want, dtype):
        tol = 2e-2 if dtype == torch.bfloat16 else 1e-5
        self.assertTrue(bool(torch.isfinite(got).all()))
        mag = want.abs().max()
        if mag == 0:
            self.assertEqual(float(got.abs().max()), 0.0)
            return
        err = (got.double() - want).abs().max() / mag
        self.assertLess(float(err), tol)

    def run_full(
        self,
        lens,
        causal=True,
        targets=None,
        window=0,
        ctx=0,
        attn_scale=None,
        dtype=torch.bfloat16,
        fused=False,
        min_full=0,
    ):
        q, k, v, offsets = self.make(lens, dtype, fused)
        num_targets = None if targets is None else torch.tensor(targets)
        max_seq_len = max(lens) + 3
        alpha = 1.0 / math.sqrt(self.D)
        got = hstu_mha(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            offsets,
            causal,
            num_targets,
            attn_scale,
            window,
            ctx,
            min_full,
        )
        self.assertEqual(got.shape, (q.shape[0], self.H, self.V))
        self.assertEqual(got.dtype, dtype)
        want = reference(
            max_seq_len,
            alpha,
            q,
            k,
            v,
            offsets,
            causal,
            num_targets,
            attn_scale,
            window,
            ctx,
            min_full_attn_seq_len=min_full,
        )
        self.check(got, want, dtype)

    def test_causal(self):
        self.run_full([5, 130, 64])

    def test_targets(self):
        self.run_full([50, 200, 129], targets=[5, 0, 20])

    def test_window(self):
        self.run_full([300, 129], targets=[7, 3], window=17)

    def test_window_min_full(self):
        self.run_full([300, 129, 700], targets=[7, 3, 40], window=17, min_full=9)

    def test_long_window(self):
        self.run_full([1100, 700], targets=[64, 10], window=300)

    def test_contextual(self):
        self.run_full([90, 260], targets=[6, 31], ctx=3)

    def test_noncausal(self):
        self.run_full([70, 33], causal=False)

    def test_float32(self):
        self.run_full([40, 77], targets=[3, 9], dtype=torch.float32)

    def test_fused_uvqk_views(self):
        self.run_full([256, 300], targets=[32, 16], fused=True)

    def test_attn_scale(self):
        lens = [77, 140]
        # Every scalar is used as given, including 0 and negative values.
        for value in (0.01, 0.37, 0.0, -0.5):
            self.run_full(lens, targets=[5, 0], attn_scale=torch.tensor(value))
        self.run_full(
            lens, targets=[5, 0], attn_scale=torch.rand(sum(lens)) * 0.02 + 0.005
        )

    def run_delta(self, lens, delta, targets, windows, ctx=0):
        q, k, v, offsets = self.make(lens)
        rows = torch.cat(
            [
                torch.arange(int(offsets[b + 1]) - delta, int(offsets[b + 1]))
                for b in range(len(lens))
            ]
        )
        dq = q[rows]
        num_targets = torch.tensor(targets)
        alpha = 1.0 / math.sqrt(self.D)
        max_seq_len = max(lens) + 3
        for window in windows:
            got = delta_hstu_mha(
                max_seq_len, alpha, dq, k, v, offsets, num_targets, window, ctx
            )
            want = reference(
                max_seq_len,
                alpha,
                dq,
                k,
                v,
                offsets,
                True,
                num_targets,
                None,
                window,
                ctx,
                delta=delta,
            )
            self.check(got, want, torch.bfloat16)

    def test_delta(self):
        self.run_delta([40, 300, 129], 16, [16, 4, 0], (0, 25))

    def test_delta_multi_tile(self):
        # Several query tiles per sequence (delta > 32, not a multiple of 32).
        self.run_delta([700, 1100, 650], 200, [50, 600, 10], (0, 300))
        self.run_delta([700, 1100, 650], 600, [600, 100, 0], (0, 300))

    def test_delta_contextual(self):
        self.run_delta([90, 260, 300], 40, [6, 31, 0], (0, 20), ctx=3)

    def test_meta_shapes(self):
        q, k, v, offsets = self.make([10, 20])
        mode = torch._subclasses.fake_tensor.FakeTensorMode()
        fq, fk, fv, fo = (mode.from_tensor(t) for t in (q, k, v, offsets))
        with mode:
            out = hstu_mha(32, 0.1, fq, fk, fv, fo)
            self.assertEqual(out.shape, (30, self.H, self.V))
            dout = delta_hstu_mha(32, 0.1, fq[:4], fk, fv, fo)
            self.assertEqual(dout.shape, (4, self.H, self.V))

    def test_compile_dynamic_lengths(self):
        # max_seq_len is a SymInt: new lengths must not specialize the graph.
        from torch._dynamo.testing import CompileCounter

        counter = CompileCounter()

        def fn(q, k, v, offsets, targets, max_seq_len):
            return hstu_mha(max_seq_len, 0.1, q, k, v, offsets, True, targets)

        compiled = torch.compile(fn, backend=counter, dynamic=True)
        for n in (40, 57, 73):
            q, k, v, offsets = self.make([n])
            out = compiled(q, k, v, offsets, torch.tensor([4]), n)
            self.assertEqual(out.shape, (n, self.H, self.V))
        self.assertEqual(counter.frame_count, 1)

    def test_compile_inductor_matches_eager(self):
        def fn(q, k, v, offsets, targets, max_seq_len):
            full = hstu_mha(max_seq_len, 0.1, q, k, v, offsets, True, targets)
            delta = delta_hstu_mha(
                max_seq_len, 0.1, q[-4:], k, v, offsets[[0, -1]], targets[-1:]
            )
            return full, delta

        compiled = torch.compile(fn, backend="inductor", dynamic=True, fullgraph=True)
        for n in (40, 57):
            q, k, v, offsets = self.make([n])
            targets = torch.tensor([4])
            got = compiled(q, k, v, offsets, targets, n)
            want = fn(q, k, v, offsets, targets, n)
            for g, w in zip(got, want, strict=True):
                self.assertEqual(g, w)

    def test_keeps_thread_count(self):
        q, k, v, offsets = self.make([10, 20])
        before = torch.get_num_threads()
        try:
            for n in (2, 3):
                torch.set_num_threads(n)
                hstu_mha(32, 0.1, q, k, v, offsets)
                delta_hstu_mha(32, 0.1, q[:4], k, v, offsets)
                self.assertEqual(torch.get_num_threads(), n)
        finally:
            torch.set_num_threads(before)

    def test_compile_with_requires_grad(self):
        q, k, v, offsets = self.make([10, 20])
        q.requires_grad_(True)

        def fn(q, k, v, offsets):
            return hstu_mha(32, 0.1, q, k, v, offsets)

        out = torch.compile(fn, backend="inductor")(q, k, v, offsets)
        self.assertEqual(out.detach(), fn(q, k, v, offsets).detach())
        with self.assertRaises(RuntimeError):
            out.float().sum().backward()

    def test_backward_raises(self):
        q, k, v, offsets = self.make([10, 20])
        q.requires_grad_(True)
        out = hstu_mha(32, 0.1, q, k, v, offsets)
        with self.assertRaises(RuntimeError):
            out.float().sum().backward()

    def test_invalid(self):
        q, k, v, offsets = self.make([10, 20])
        buf = torch.empty_like(v)

        def op(*args, **kwargs):
            kwargs.setdefault("out", buf)
            return torch.ops.zentorch.zentorch_hstu_mha.out(*args, **kwargs)

        def delta_op(*args, **kwargs):
            kwargs.setdefault("out", buf)
            return torch.ops.zentorch.zentorch_delta_hstu_mha.out(*args, **kwargs)

        cases = {
            "heads mismatch": lambda: op(32, 0.1, q, k[:, :1], v, offsets),
            "length > max_seq_len": lambda: op(15, 0.1, q, k, v, offsets),
            "q rows != k rows": lambda: op(32, 0.1, q[:20], k, v, offsets),
            "offsets do not cover q": lambda: op(
                32, 0.1, q, k, v, torch.tensor([0, 10, 25])
            ),
            "offsets do not start at 0": lambda: op(
                32, 0.1, q, k, v, torch.tensor([5, 10, 30])
            ),
            "decreasing offsets": lambda: op(
                32, 0.1, q, k, v, torch.tensor([0, 20, 10, 30])
            ),
            "float offsets": lambda: op(32, 0.1, q, k, v, offsets.float()),
            "targets > length": lambda: op(
                32, 0.1, q, k, v, offsets, True, torch.tensor([2, 21])
            ),
            "float targets": lambda: op(
                32, 0.1, q, k, v, offsets, True, torch.tensor([2.0, 3.0])
            ),
            "attn_scale shape": lambda: op(
                32, 0.1, q, k, v, offsets, True, None, torch.ones(5)
            ),
            "mixed dtypes": lambda: op(32, 0.1, q, k.float(), v, offsets),
            "head dim not contiguous": lambda: op(
                32, 0.1, q, k.transpose(1, 2).contiguous().transpose(1, 2), v, offsets
            ),
            "broadcast k": lambda: op(32, 0.1, q, k[:1].expand_as(k), v, offsets),
            "delta rows not a multiple": lambda: delta_op(
                32, 0.1, q[:5], k, v, offsets
            ),
            "delta longer than a sequence": lambda: delta_op(
                32, 0.1, q[:24], k, v, offsets
            ),
            "negative window": lambda: op(
                32, 0.1, q, k, v, offsets, True, None, None, -1
            ),
            "negative contextual": lambda: op(
                32, 0.1, q, k, v, offsets, True, None, None, 0, -1
            ),
            "negative min_full": lambda: op(
                32, 0.1, q, k, v, offsets, True, None, None, 4, 0, -1
            ),
            "empty head dim": lambda: op(32, 0.1, q[..., :0], k[..., :0], v, offsets),
            "out shape": lambda: op(
                32,
                0.1,
                q,
                k,
                v,
                offsets,
                out=torch.empty(q.shape[0], self.H, self.D, dtype=q.dtype),
            ),
            "out dtype": lambda: op(
                32, 0.1, q, k, v, offsets, out=torch.empty_like(v, dtype=torch.float32)
            ),
            "out head dim not contiguous": lambda: op(
                32,
                0.1,
                q,
                k,
                v,
                offsets,
                out=buf.transpose(1, 2).contiguous().transpose(1, 2),
            ),
        }
        for name, call in cases.items():
            with self.subTest(name), self.assertRaises(RuntimeError):
                call()

    def test_empty(self):
        q, k, v, offsets = self.make([])
        out = hstu_mha(32, 0.1, q, k, v, offsets)
        self.assertEqual(out.shape, (0, self.H, self.V))
        dout = delta_hstu_mha(32, 0.1, q, k, v, offsets)
        self.assertEqual(dout.shape, (0, self.H, self.V))


if __name__ == "__main__":
    run_tests()
