# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for ``chunk_local_cumsum``."""

from __future__ import annotations

import sys
import unittest
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F
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
SEQLENS_VARLEN = [[64], [65], [7], [64, 64], [70, 30, 100]]
SEQLENS_QWEN = [[64], [128, 64], [200, 100, 50]]


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


def _kernel_reverse_cumsum(chunk: torch.Tensor, time_dim: int) -> torch.Tensor:
    fwd = chunk.cumsum(time_dim)
    total = chunk.sum(time_dim, keepdim=True)
    return -fwd + total + chunk


def chunk_local_cumsum_oracle(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    **kwargs,
) -> torch.Tensor:
    """Independent oracle for ``chunk_local_cumsum``."""
    if not _is_power_of_two(chunk_size):
        raise ValueError(f"chunk_size must be a power of 2; got {chunk_size}")
    BT = chunk_size

    if g.dim() == 3:
        if head_first:
            B, H, T = g.shape
            time_dim = 2
        else:
            B, T, H = g.shape
            time_dim = 1
    elif g.dim() == 4:
        if head_first:
            B, H, T, _S = g.shape
            time_dim = 2
        else:
            B, T, H, _S = g.shape
            time_dim = 1
    else:
        raise ValueError(f"g must be 3-D or 4-D; got shape {tuple(g.shape)}")

    out_dtype = output_dtype if output_dtype is not None else g.dtype
    g_f = g.float()

    if cu_seqlens is None:
        NT = (T + BT - 1) // BT
        pad = NT * BT - T
        if pad > 0:
            n_dims = g_f.dim()
            pad_list = [0] * (2 * n_dims)
            pair_idx = (n_dims - 1 - time_dim) * 2 + 1
            pad_list[pair_idx] = pad
            g_padded = F.pad(g_f, pad_list)
        else:
            g_padded = g_f
        new_shape = list(g_padded.shape)
        new_shape[time_dim : time_dim + 1] = [NT, BT]
        g_chunks = g_padded.reshape(new_shape)
        if reverse:
            o_chunks = _kernel_reverse_cumsum(g_chunks, time_dim=time_dim + 1)
        else:
            o_chunks = g_chunks.cumsum(dim=time_dim + 1)
        o_padded = o_chunks.reshape(g_padded.shape)
        return o_padded.narrow(time_dim, 0, T).to(out_dtype).contiguous()

    if B != 1:
        raise ValueError(f"Only batch size 1 is supported with cu_seqlens; got B={B}")
    if chunk_indices is None:
        chunk_indices = _oracle_prepare_chunk_indices(cu_seqlens, BT)

    o = torch.empty_like(g, dtype=out_dtype)
    for i in range(chunk_indices.shape[0]):
        seq_idx = int(chunk_indices[i, 0].item())
        chunk_idx = int(chunk_indices[i, 1].item())
        bos = int(cu_seqlens[seq_idx].item())
        eos = int(cu_seqlens[seq_idx + 1].item())
        cs_start = bos + chunk_idx * BT
        cs_end = min(cs_start + BT, eos)
        chunk_len = cs_end - cs_start
        if chunk_len <= 0:
            continue

        chunk_slice = g_f.narrow(0, 0, 1).narrow(time_dim, cs_start, chunk_len)
        running = torch.zeros_like(chunk_slice.narrow(time_dim, 0, 1))
        out_slice = torch.empty_like(chunk_slice)
        if reverse:
            for t in range(chunk_len - 1, -1, -1):
                cur = chunk_slice.narrow(time_dim, t, 1)
                running = running + cur
                out_slice.narrow(time_dim, t, 1).copy_(running)
        else:
            for t in range(chunk_len):
                cur = chunk_slice.narrow(time_dim, t, 1)
                running = running + cur
                out_slice.narrow(time_dim, t, 1).copy_(running)

        o.narrow(0, 0, 1).narrow(time_dim, cs_start, chunk_len).copy_(
            out_slice.to(out_dtype)
        )
    return o


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _seqlens_name(seqlens: list[int]) -> str:
    return "x".join(map(str, seqlens))


def _name_dtype_seqlens(method, _idx, params):
    dtype, seqlens = params.args
    return f"{method.__name__}_{_dtype_name(dtype)}_{_seqlens_name(seqlens)}"


def _make_tensor(
    shape: tuple[int, ...], dtype: torch.dtype, device: torch.device, seed: int = 0
) -> torch.Tensor:
    g = torch.Generator(device=device).manual_seed(seed)
    return torch.randn(*shape, dtype=dtype, device=device, generator=g) * 0.5


def _make_cu_seqlens(seqlens: list[int], device: torch.device) -> torch.Tensor:
    qsl = [0]
    for sl in seqlens:
        qsl.append(qsl[-1] + sl)
    return torch.tensor(qsl, dtype=torch.int32, device=device)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GDN_ChunkLocalCumsum(Zentorch_TestCase):

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
        H = 4
        chunk_size = 64
        T_total = sum(seqlens)
        g = _make_tensor((1, T_total, H), dtype, cpu, seed=T_total + H)
        cu_seqlens = _make_cu_seqlens(seqlens, cpu)
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

        o_cpp = torch.ops.zentorch.gdn_chunk_local_cumsum(
            g, chunk_size, cu_seqlens, chunk_indices,
        )
        o_oracle = chunk_local_cumsum_oracle(
            g, chunk_size=chunk_size, cu_seqlens=cu_seqlens,
        )
        atol, rtol = default_tolerance(o_oracle.dtype)
        self.assertEqual(o_cpp.float(), o_oracle, atol=atol, rtol=rtol)

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
        chunk_size = 64
        T_total = sum(seqlens)
        g = _make_tensor((1, T_total, s.num_v_heads), dtype, cpu, seed=T_total)
        cu_seqlens = _make_cu_seqlens(seqlens, cpu)
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)

        o_cpp = torch.ops.zentorch.gdn_chunk_local_cumsum(
            g, chunk_size, cu_seqlens, chunk_indices,
        )
        o_oracle = chunk_local_cumsum_oracle(
            g, chunk_size=chunk_size, cu_seqlens=cu_seqlens,
        )
        atol, rtol = default_tolerance(o_oracle.dtype)
        self.assertEqual(o_cpp.float(), o_oracle, atol=atol, rtol=rtol)


if __name__ == "__main__":
    run_tests()
