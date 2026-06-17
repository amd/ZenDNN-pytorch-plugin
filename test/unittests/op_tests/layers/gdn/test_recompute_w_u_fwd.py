# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for ``recompute_w_u_fwd``."""

from __future__ import annotations

import sys
import unittest
from itertools import product
from pathlib import Path

import torch
from parameterized import parameterized

from test.unittests.op_tests.layers.gdn.helpers.shapes import Qwen35_4B_GDN
from test.unittests.op_tests.layers.gdn.helpers.varlen import prepare_chunk_indices

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from unittest_utils import (  # noqa: E402
    Zentorch_TestCase,
    default_tolerance,
    has_zentorch,
    run_tests,
)


DTYPES = [torch.float32, torch.bfloat16, torch.float16]
SEQLENS_VARLEN = [[64], [70, 30, 100]]
SEQLENS_QWEN = [[64], [128, 64]]


def _oracle_prepare_chunk_indices(
    cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    n_chunks = (lens + chunk_size - 1) // chunk_size
    chunk_in_seq = torch.cat(
        [torch.arange(int(n.item()), device=cu_seqlens.device) for n in n_chunks]
    )
    seq_idx = chunk_in_seq.eq(0).cumsum(0) - 1
    return torch.stack([seq_idx, chunk_in_seq], dim=1).to(cu_seqlens)


def _chunk_ranges(
    *, B: int, T: int, BT: int,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
) -> list[tuple[int, int, int]]:
    if cu_seqlens is None:
        return [
            (b, c * BT, min((c + 1) * BT, T))
            for b in range(B)
            for c in range((T + BT - 1) // BT)
        ]
    if B != 1:
        raise ValueError(f"Varlen mode requires B == 1; got {B}")
    if chunk_indices is None:
        chunk_indices = _oracle_prepare_chunk_indices(cu_seqlens, BT)
    out: list[tuple[int, int, int]] = []
    for i in range(chunk_indices.shape[0]):
        seq_idx = int(chunk_indices[i, 0].item())
        chunk_idx = int(chunk_indices[i, 1].item())
        bos = int(cu_seqlens[seq_idx].item())
        eos = int(cu_seqlens[seq_idx + 1].item())
        cs_start = bos + chunk_idx * BT
        cs_end = min(cs_start + BT, eos)
        out.append((0, cs_start, cs_end))
    return out


def recompute_w_u_fwd_oracle(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent oracle for ``recompute_w_u_fwd``."""
    B, T, Hg, K_dim = k.shape
    _, _, H, V_dim = v.shape
    BT = A.shape[-1]
    r = H // Hg

    w = torch.zeros(B, T, H, K_dim, dtype=k.dtype, device=k.device)
    u = torch.zeros(B, T, H, V_dim, dtype=v.dtype, device=v.device)

    chunks = _chunk_ranges(
        B=B, T=T, BT=BT, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    for b, start, end in chunks:
        BT_eff = end - start
        if BT_eff <= 0:
            continue
        for h in range(H):
            kh = h // r
            A_block = A[b, start:end, h, :BT_eff].float()
            v_block = v[b, start:end, h].float()
            k_block = k[b, start:end, kh].float()
            beta_block = beta[b, start:end, h].float()
            g_block = g_cumsum[b, start:end, h].float()

            vb = v_block * beta_block[..., None]
            u_block = torch.einsum("ij,jk->ik", A_block, vb)
            u[b, start:end, h, :] = u_block.to(v.dtype)

            gate_factor = beta_block * torch.exp(g_block)
            kb = k_block * gate_factor[..., None]
            w_block = torch.einsum("ij,jk->ik", A_block, kb)
            w[b, start:end, h, :] = w_block.to(k.dtype)

    return w, u


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _seqlens_name(seqlens: list[int]) -> str:
    return "x".join(map(str, seqlens))


def _name_dtype_seqlens(method, _idx, params):
    dtype, seqlens = params.args
    return f"{method.__name__}_{_dtype_name(dtype)}_{_seqlens_name(seqlens)}"


def _build_inputs(
    *,
    B: int, T: int,
    Hg: int, H: int,
    K_dim: int, V_dim: int,
    BT: int,
    dtype: torch.dtype,
    device: torch.device,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(B * 1000 + T + H + K_dim + seed)
    k = torch.randn(B, T, Hg, K_dim, dtype=dtype, device=device, generator=g) * 0.5
    v = torch.randn(B, T, H, V_dim, dtype=dtype, device=device, generator=g) * 0.5
    beta = torch.randn(B, T, H, dtype=dtype, device=device, generator=g) * 0.3 + 0.5
    g_cumsum = torch.randn(B, T, H, dtype=torch.float32, device=device, generator=g) * 0.3
    A = torch.randn(B, T, H, BT, dtype=dtype, device=device, generator=g) * 0.4
    return {"k": k, "v": v, "beta": beta, "g_cumsum": g_cumsum, "A": A}


def _cu_seqlens(seqlens: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [0] + [sum(seqlens[: i + 1]) for i in range(len(seqlens))],
        dtype=torch.int32,
        device=device,
    )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GDN_RecomputeWUFwd(Zentorch_TestCase):

    @parameterized.expand(
        list(product(DTYPES, SEQLENS_VARLEN)),
        name_func=_name_dtype_seqlens,
    )
    def test_cpp_matches_oracle_varlen(
        self,
        dtype: torch.dtype,
        seqlens: list[int],
    ) -> None:
        cpu = torch.device("cpu")
        BT = 64
        Hg, H, K_dim, V_dim = 4, 8, 32, 32
        T_total = sum(seqlens)
        inputs = _build_inputs(
            B=1, T=T_total, Hg=Hg, H=H, K_dim=K_dim, V_dim=V_dim, BT=BT,
            dtype=dtype, device=cpu,
        )
        cu = _cu_seqlens(seqlens, cpu)
        chunk_indices = prepare_chunk_indices(cu, BT)

        w_cpp, u_cpp = torch.ops.zentorch.gdn_recompute_w_u_fwd(
            inputs["k"], inputs["v"], inputs["beta"],
            inputs["g_cumsum"], inputs["A"],
            cu, chunk_indices,
        )
        w_oracle, u_oracle = recompute_w_u_fwd_oracle(
            k=inputs["k"], v=inputs["v"], beta=inputs["beta"],
            g_cumsum=inputs["g_cumsum"], A=inputs["A"],
            cu_seqlens=cu,
        )
        atol, rtol = default_tolerance(dtype)
        self.assertEqual(w_cpp, w_oracle, atol=atol, rtol=rtol, msg="w mismatch")
        self.assertEqual(u_cpp, u_oracle, atol=atol, rtol=rtol, msg="u mismatch")

    @parameterized.expand(
        [(torch.bfloat16, sl) for sl in SEQLENS_QWEN],
        name_func=_name_dtype_seqlens,
    )
    def test_qwen35_4b_call_site_shape(
        self,
        dtype: torch.dtype,
        seqlens: list[int],
    ) -> None:
        s = Qwen35_4B_GDN
        cpu = torch.device("cpu")
        BT = 64
        T_total = sum(seqlens)
        inputs = _build_inputs(
            B=1, T=T_total, Hg=s.num_k_heads, H=s.num_v_heads,
            K_dim=s.head_k_dim, V_dim=s.head_v_dim, BT=BT,
            dtype=dtype, device=cpu,
        )
        cu = _cu_seqlens(seqlens, cpu)
        chunk_indices = prepare_chunk_indices(cu, BT)

        w_cpp, u_cpp = torch.ops.zentorch.gdn_recompute_w_u_fwd(
            inputs["k"], inputs["v"], inputs["beta"],
            inputs["g_cumsum"], inputs["A"],
            cu, chunk_indices,
        )
        w_oracle, u_oracle = recompute_w_u_fwd_oracle(
            k=inputs["k"], v=inputs["v"], beta=inputs["beta"],
            g_cumsum=inputs["g_cumsum"], A=inputs["A"],
            cu_seqlens=cu,
        )
        atol, rtol = default_tolerance(dtype)
        self.assertEqual(w_cpp, w_oracle, atol=atol, rtol=rtol, msg="w mismatch")
        self.assertEqual(u_cpp, u_oracle, atol=atol, rtol=rtol, msg="u mismatch")


if __name__ == "__main__":
    run_tests()
