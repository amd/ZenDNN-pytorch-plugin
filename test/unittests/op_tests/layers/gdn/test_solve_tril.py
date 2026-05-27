# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for ``solve_tril``."""

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
ALLOWED_BTS = [16, 32, 64]
SEQLENS_VARLEN = [
    [16],
    [64],
    [70],
    [32, 64],
    [80, 16, 50],
]
SEQLENS_QWEN = [[64], [128, 64]]

_ALLOWED_BT = (16, 32, 64)


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


def _neumann_inverse(A_strict: torch.Tensor) -> torch.Tensor:
    """``(I + A_strict)^{-1}`` via the truncated Neumann series."""
    n = A_strict.shape[-1]
    M = torch.eye(n, dtype=A_strict.dtype, device=A_strict.device)
    neg_A = -A_strict
    neg_A_pow = neg_A.clone()
    for k in range(1, n):
        M = M + neg_A_pow
        if k == n - 1:
            break
        neg_A_pow = neg_A_pow @ neg_A
        if neg_A_pow.abs().max() == 0:
            break
    return M


def solve_tril_oracle(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = torch.float,
) -> torch.Tensor:
    """Independent oracle for ``solve_tril``."""
    if A.dim() != 4:
        raise ValueError(f"A must be 4-D; got shape {tuple(A.shape)}")
    B, T, H, BT = A.shape
    if BT not in _ALLOWED_BT:
        raise ValueError(f"BT must be one of {_ALLOWED_BT}; got {BT}")
    if output_dtype is None:
        output_dtype = A.dtype

    Ai = torch.zeros_like(A, dtype=output_dtype)
    chunks = _chunk_ranges(
        B=B, T=T, BT=BT, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices,
    )
    for b, start, end in chunks:
        BT_eff = end - start
        if BT_eff <= 0:
            continue
        for h in range(H):
            A_block = A[b, start:end, h, :BT_eff].float()
            A_strict = torch.tril(A_block, diagonal=-1)
            M = _neumann_inverse(A_strict)
            Ai[b, start:end, h, :BT_eff] = M.to(output_dtype)
    return Ai


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _seqlens_name(seqlens: list[int]) -> str:
    return "x".join(map(str, seqlens))


def _name_dtype_bt_seqlens(method, _idx, params):
    dtype, BT, seqlens = params.args
    return (
        f"{method.__name__}_{_dtype_name(dtype)}_"
        f"bt{BT}_{_seqlens_name(seqlens)}"
    )


def _name_dtype_seqlens(method, _idx, params):
    dtype, seqlens = params.args
    return f"{method.__name__}_{_dtype_name(dtype)}_{_seqlens_name(seqlens)}"


def _make_strict_lower_A(
    *,
    B: int, T: int, H: int, BT: int,
    dtype: torch.dtype, device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(B * 1000 + T + H + BT + seed)
    A = torch.zeros(B, T, H, BT, dtype=dtype, device=device)
    NT_full = T // BT
    leftover = T % BT
    BT_eff_for_partial = leftover

    for b in range(B):
        for c in range(NT_full + (1 if leftover > 0 else 0)):
            BT_eff = BT if c < NT_full else BT_eff_for_partial
            if BT_eff == 0:
                continue
            for h in range(H):
                M = torch.randn(BT_eff, BT_eff, dtype=dtype, device=device, generator=g) * 0.2
                M = torch.tril(M, diagonal=-1)
                A[b, c * BT : c * BT + BT_eff, h, :BT_eff] = M
    return A


def _cu_seqlens(seqlens: list[int], device: torch.device) -> torch.Tensor:
    qsl = [0]
    for sl in seqlens:
        qsl.append(qsl[-1] + sl)
    return torch.tensor(qsl, dtype=torch.int32, device=device)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GDN_SolveTril(Zentorch_TestCase):

    @parameterized.expand(
        list(product(DTYPES, ALLOWED_BTS, SEQLENS_VARLEN)),
        name_func=_name_dtype_bt_seqlens,
    )
    def test_cpp_matches_oracle_varlen(
        self,
        dtype: torch.dtype,
        BT: int,
        seqlens: list[int],
    ) -> None:
        cpu = torch.device("cpu")
        H = 2
        T_total = sum(seqlens)
        A = _make_strict_lower_A(B=1, T=T_total, H=H, BT=BT, dtype=dtype, device=cpu)
        cu_seqlens = _cu_seqlens(seqlens, cpu)
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

        Ai_cpp = torch.ops.zentorch.gdn_solve_tril(A, cu_seqlens, chunk_indices)
        Ai_oracle = solve_tril_oracle(
            A, cu_seqlens=cu_seqlens, output_dtype=torch.float32,
        )
        atol, rtol = default_tolerance(dtype)
        self.assertEqual(
            Ai_cpp.to(Ai_oracle.dtype), Ai_oracle, atol=atol, rtol=rtol,
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
        BT = 64
        A = _make_strict_lower_A(
            B=1, T=T_total, H=s.num_v_heads, BT=BT, dtype=dtype, device=cpu,
        )
        cu_seqlens = _cu_seqlens(seqlens, cpu)
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)

        Ai_cpp = torch.ops.zentorch.gdn_solve_tril(A, cu_seqlens, chunk_indices)
        Ai_oracle = solve_tril_oracle(
            A, cu_seqlens=cu_seqlens, output_dtype=torch.float32,
        )
        atol, rtol = default_tolerance(dtype)
        self.assertEqual(
            Ai_cpp.to(Ai_oracle.dtype), Ai_oracle, atol=atol, rtol=rtol,
        )


if __name__ == "__main__":
    run_tests()
