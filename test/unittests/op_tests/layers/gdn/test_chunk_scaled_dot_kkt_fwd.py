# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for ``chunk_scaled_dot_kkt_fwd``."""

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


DTYPES = [torch.float32, torch.bfloat16]
SEQLENS_VARLEN = [[64], [70, 30, 100]]
SEQLENS_QWEN = [[64], [128, 64]]


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


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


def chunk_scaled_dot_kkt_fwd_oracle(
    k: torch.Tensor,
    g: torch.Tensor | None = None,
    beta: torch.Tensor | None = None,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Independent oracle for ``chunk_scaled_dot_kkt_fwd``."""
    if not _is_power_of_two(chunk_size):
        raise ValueError(f"chunk_size must be a power of 2; got {chunk_size}")
    if beta is None:
        raise ValueError(
            "beta is required; oracle uses beta.shape and beta[...] internally"
        )
    B, T, Hg, K_dim = k.shape
    H = beta.shape[-1]
    r = H // Hg
    BT = chunk_size

    A = torch.zeros(B, T, H, BT, device=k.device, dtype=output_dtype)
    chunks = _chunk_ranges(
        B=B, T=T, BT=BT, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    for b, start, end in chunks:
        chunk_len = end - start
        if chunk_len <= 0:
            continue
        idx = torch.arange(chunk_len, device=k.device)
        strict_lower = idx.unsqueeze(-1) > idx.unsqueeze(0)

        for h in range(H):
            kh = h // r
            k_chunk = k[b, start:end, kh].float()
            beta_chunk = beta[b, start:end, h].float()
            kb = k_chunk * beta_chunk[..., None]
            A_chunk = torch.einsum("ik,jk->ij", kb, k_chunk)

            if g is not None:
                g_chunk = g[b, start:end, h].float()
                decay = torch.exp(g_chunk[..., None] - g_chunk[None, :])
                A_chunk = A_chunk * decay

            A_chunk = A_chunk * strict_lower.to(A_chunk.dtype)
            A[b, start:end, h, :chunk_len] = A_chunk.to(output_dtype)
    return A


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _seqlens_name(seqlens: list[int]) -> str:
    return "x".join(map(str, seqlens))


def _name_dtype_seqlens(method, _idx, params):
    dtype, seqlens = params.args
    return f"{method.__name__}_{_dtype_name(dtype)}_{_seqlens_name(seqlens)}"


def _build_inputs(
    *,
    B: int,
    T: int,
    Hg: int,
    H: int,
    K_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    with_g: bool = True,
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(B * 1000 + T + H + K_dim)
    k = torch.randn(B, T, Hg, K_dim, dtype=dtype, device=device, generator=g) * 0.5
    beta = torch.randn(B, T, H, dtype=dtype, device=device, generator=g) * 0.3 + 0.5
    g_tensor = (
        torch.randn(B, T, H, dtype=torch.float32, device=device, generator=g) * 0.2
        if with_g else None
    )
    return {"k": k, "g": g_tensor, "beta": beta}


def _cu_seqlens(seqlens: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [0] + [sum(seqlens[: i + 1]) for i in range(len(seqlens))],
        dtype=torch.int32,
        device=device,
    )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GDN_ChunkScaledDotKktFwd(Zentorch_TestCase):

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
        Hg, H, K_dim = 4, 8, 32
        T_total = sum(seqlens)
        inputs = _build_inputs(
            B=1, T=T_total, Hg=Hg, H=H, K_dim=K_dim,
            dtype=dtype, device=cpu, with_g=True,
        )
        cu = _cu_seqlens(seqlens, cpu)
        chunk_indices = prepare_chunk_indices(cu, 64)

        A_cpp = torch.ops.zentorch.gdn_chunk_scaled_dot_kkt_fwd(
            inputs["k"], inputs["g"], inputs["beta"], cu, chunk_indices, 64,
        )
        A_oracle = chunk_scaled_dot_kkt_fwd_oracle(
            k=inputs["k"], g=inputs["g"], beta=inputs["beta"],
            cu_seqlens=cu, chunk_size=64,
        )
        atol, rtol = default_tolerance(dtype, torch.float32)
        self.assertEqual(
            A_cpp.to(A_oracle.dtype), A_oracle, atol=atol, rtol=rtol,
        )

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
        T_total = sum(seqlens)
        inputs = _build_inputs(
            B=1, T=T_total, Hg=s.num_k_heads, H=s.num_v_heads,
            K_dim=s.head_k_dim, dtype=dtype, device=cpu, with_g=True,
        )
        cu = _cu_seqlens(seqlens, cpu)
        chunk_indices = prepare_chunk_indices(cu, 64)

        A_cpp = torch.ops.zentorch.gdn_chunk_scaled_dot_kkt_fwd(
            inputs["k"], inputs["g"], inputs["beta"], cu, chunk_indices, 64,
        )
        A_oracle = chunk_scaled_dot_kkt_fwd_oracle(
            k=inputs["k"], g=inputs["g"], beta=inputs["beta"],
            cu_seqlens=cu, chunk_size=64,
        )
        atol, rtol = default_tolerance(dtype, torch.float32)
        self.assertEqual(
            A_cpp.to(A_oracle.dtype), A_oracle, atol=atol, rtol=rtol,
        )


if __name__ == "__main__":
    run_tests()
