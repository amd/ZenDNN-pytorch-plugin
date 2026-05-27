# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for ``chunk_fwd_o``."""

from __future__ import annotations

import sys
import unittest
from itertools import product
from pathlib import Path

import torch
from parameterized import parameterized

from test.unittests.op_tests.layers.gdn.helpers.shapes import Qwen35_4B_GDN
from test.unittests.op_tests.layers.gdn.helpers.varlen import prepare_chunk_offsets

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from unittest_utils import (  # noqa: E402
    Zentorch_TestCase,
    default_tolerance,
    has_zentorch,
    run_tests,
)


DTYPES = [torch.float32, torch.bfloat16]
SEQLENS_VARLEN = [[64], [70, 30, 100]]
SEQLENS_QWEN = [[64], [128, 64]]


def _oracle_prepare_chunk_offsets(
    cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    lens = cu_seqlens[1:] - cu_seqlens[:-1]
    n_chunks = (lens + chunk_size - 1) // chunk_size
    return torch.cat(
        [cu_seqlens.new_zeros(1), n_chunks.to(cu_seqlens.dtype)]
    ).cumsum(0)


def chunk_fwd_o_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """Independent oracle for ``chunk_fwd_o``."""
    B, T, Hg, K_dim = q.shape
    _, _, H, V_dim = v.shape
    r = H // Hg
    BT = chunk_size

    if scale is None:
        scale = K_dim ** -0.5

    device = v.device
    o = torch.zeros(B, T, H, V_dim, dtype=v.dtype, device=device)

    if cu_seqlens is None:
        N = B
        NT_per_seq = (T + BT - 1) // BT
        seq_chunk_starts = [0] * B
        seq_chunk_counts = [NT_per_seq] * B
    else:
        N = cu_seqlens.numel() - 1
        chunk_offsets = _oracle_prepare_chunk_offsets(cu_seqlens, BT)
        seq_chunk_starts = [int(chunk_offsets[n].item()) for n in range(N)]
        seq_chunk_counts = [
            int(chunk_offsets[n + 1].item()) - int(chunk_offsets[n].item())
            for n in range(N)
        ]

    for n in range(N):
        if cu_seqlens is None:
            in_batch_idx = n
            chunk_offset_in_seq = 0
            T_n = T
        else:
            in_batch_idx = 0
            chunk_offset_in_seq = int(cu_seqlens[n].item())
            T_n = int(cu_seqlens[n + 1].item()) - chunk_offset_in_seq

        boh = seq_chunk_starts[n]
        chunks_in_seq = seq_chunk_counts[n]
        h_batch_idx = n if cu_seqlens is None else 0

        for h_idx in range(H):
            kh = h_idx // r
            for i_t in range(chunks_in_seq):
                chunk_start = chunk_offset_in_seq + i_t * BT
                chunk_end = min(chunk_start + BT, chunk_offset_in_seq + T_n)
                BT_eff = chunk_end - chunk_start
                if BT_eff <= 0:
                    continue

                q_block = q[in_batch_idx, chunk_start:chunk_end, kh].float()
                k_block = k[in_batch_idx, chunk_start:chunk_end, kh].float()
                v_block = v[in_batch_idx, chunk_start:chunk_end, h_idx].float()
                h_chunk = h[h_batch_idx, boh + i_t, h_idx].float()

                o_history = torch.einsum("tk,vk->tv", q_block, h_chunk)
                A = torch.einsum("tk,jk->tj", q_block, k_block)

                if g is not None:
                    g_block = g[in_batch_idx, chunk_start:chunk_end, h_idx].float()
                    o_history = o_history * torch.exp(g_block)[..., None]
                    decay = torch.exp(g_block[..., None] - g_block[None, :])
                    A = A * decay

                idx = torch.arange(BT_eff, device=device)
                causal_bool = idx[..., None] >= idx[None, :]
                A = A * causal_bool.to(A.dtype)

                o_in_chunk = torch.einsum("tj,jv->tv", A, v_block)
                o_block = (o_history + o_in_chunk) * scale
                o[in_batch_idx, chunk_start:chunk_end, h_idx] = o_block.to(v.dtype)

    return o


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
    NT_total: int,
    dtype: torch.dtype,
    device: torch.device,
    use_g: bool = True,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(B * 1000 + T + H + K_dim + seed)
    q = torch.randn(B, T, Hg, K_dim, dtype=dtype, device=device, generator=g) * 0.3
    k = torch.randn(B, T, Hg, K_dim, dtype=dtype, device=device, generator=g) * 0.3
    v = torch.randn(B, T, H, V_dim, dtype=dtype, device=device, generator=g) * 0.3
    h = torch.randn(B, NT_total, H, V_dim, K_dim, dtype=dtype, device=device, generator=g) * 0.3
    g_tensor = (
        torch.randn(B, T, H, dtype=torch.float32, device=device, generator=g) * 0.05
        if use_g else None
    )
    return {"q": q, "k": k, "v": v, "h": h, "g": g_tensor}


def _cu_seqlens(seqlens: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [0] + [sum(seqlens[: i + 1]) for i in range(len(seqlens))],
        dtype=torch.int32,
        device=device,
    )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GDN_ChunkFwdO(Zentorch_TestCase):

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
        cu = _cu_seqlens(seqlens, cpu)
        chunk_offsets = prepare_chunk_offsets(cu, BT)
        NT_total = int(chunk_offsets[-1].item())
        inputs = _build_inputs(
            B=1, T=T_total, Hg=Hg, H=H, K_dim=K_dim, V_dim=V_dim, BT=BT,
            NT_total=NT_total, dtype=dtype, device=cpu, use_g=True,
        )
        scale = K_dim ** -0.5

        o_cpp = torch.ops.zentorch.gdn_chunk_fwd_o(
            inputs["q"], inputs["k"], inputs["v"], inputs["h"], inputs["g"],
            float(scale), cu, chunk_offsets, BT,
        )
        o_oracle = chunk_fwd_o_oracle(
            **inputs, scale=scale, cu_seqlens=cu, chunk_size=BT,
        )
        atol, rtol = default_tolerance(dtype)
        self.assertEqual(o_cpp, o_oracle, atol=atol, rtol=rtol)

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
        cu = _cu_seqlens(seqlens, cpu)
        chunk_offsets = prepare_chunk_offsets(cu, BT)
        NT_total = int(chunk_offsets[-1].item())
        inputs = _build_inputs(
            B=1, T=T_total, Hg=s.num_k_heads, H=s.num_v_heads,
            K_dim=s.head_k_dim, V_dim=s.head_v_dim,
            BT=BT, NT_total=NT_total, dtype=dtype, device=cpu, use_g=True,
        )
        scale = s.head_k_dim ** -0.5

        o_cpp = torch.ops.zentorch.gdn_chunk_fwd_o(
            inputs["q"], inputs["k"], inputs["v"], inputs["h"], inputs["g"],
            float(scale), cu, chunk_offsets, BT,
        )
        o_oracle = chunk_fwd_o_oracle(
            **inputs, scale=scale, cu_seqlens=cu, chunk_size=BT,
        )
        atol, rtol = default_tolerance(dtype)
        self.assertEqual(o_cpp, o_oracle, atol=atol, rtol=rtol)


if __name__ == "__main__":
    run_tests()
