# ****************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""Varlen-metadata helpers and sentinel constants for the GDN cpp-op tests."""

from __future__ import annotations

import torch

NULL_BLOCK_ID: int = 0
PAD_SLOT_ID: int = -1


def prepare_chunk_indices(
    cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    cu = cu_seqlens.tolist()
    n_seqs = len(cu) - 1
    rows: list[tuple[int, int]] = []
    for n in range(n_seqs):
        seq_len = cu[n + 1] - cu[n]
        nt_n = (seq_len + chunk_size - 1) // chunk_size
        for c in range(nt_n):
            rows.append((n, c))
    if not rows:
        return torch.empty((0, 2), dtype=torch.int32, device=cu_seqlens.device)
    return torch.tensor(rows, dtype=torch.int32, device=cu_seqlens.device)


def prepare_chunk_offsets(
    cu_seqlens: torch.Tensor, chunk_size: int
) -> torch.Tensor:
    cu = cu_seqlens.tolist()
    n_seqs = len(cu) - 1
    out: list[int] = [0]
    for n in range(n_seqs):
        seq_len = cu[n + 1] - cu[n]
        nt_n = (seq_len + chunk_size - 1) // chunk_size
        out.append(out[-1] + nt_n)
    return torch.tensor(out, dtype=torch.int32, device=cu_seqlens.device)


__all__ = [
    "NULL_BLOCK_ID",
    "PAD_SLOT_ID",
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
]
