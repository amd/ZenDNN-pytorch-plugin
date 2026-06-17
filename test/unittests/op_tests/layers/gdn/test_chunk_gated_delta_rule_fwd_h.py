# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for ``chunk_gated_delta_rule_fwd_h``."""

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


DTYPES = [torch.float32, torch.bfloat16, torch.float16]
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


def chunk_gated_delta_rule_fwd_h_oracle(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Independent oracle for ``chunk_gated_delta_rule_fwd_h``."""
    assert gk is None, "gk not supported"

    B, T, Hg, K_dim = k.shape
    _, _, H, V_dim = u.shape
    BT = chunk_size
    r = H // Hg
    device = k.device

    if cu_seqlens is None:
        N = B
        NT_per_seq = (T + BT - 1) // BT
        NT_total = NT_per_seq
        seq_chunk_starts = [0] * B
        seq_chunk_counts = [NT_per_seq] * B
    else:
        N = cu_seqlens.numel() - 1
        if chunk_offsets is None:
            chunk_offsets = _oracle_prepare_chunk_offsets(cu_seqlens, BT)
        NT_total = int(chunk_offsets[-1].item())
        seq_chunk_starts = [int(chunk_offsets[n].item()) for n in range(N)]
        seq_chunk_counts = [
            int(chunk_offsets[n + 1].item()) - int(chunk_offsets[n].item())
            for n in range(N)
        ]

    h_out = torch.zeros(B, NT_total, H, V_dim, K_dim, dtype=k.dtype, device=device)
    v_new = (
        torch.zeros(B, T, H, V_dim, dtype=u.dtype, device=device)
        if save_new_value else None
    )
    final_state = (
        torch.zeros(N, H, V_dim, K_dim, dtype=torch.float32, device=device)
        if output_final_state else None
    )

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
            if initial_state is not None:
                state_kv = (
                    initial_state[n, h_idx].clone().float().transpose(-1, -2).contiguous()
                )
            else:
                state_kv = torch.zeros(K_dim, V_dim, dtype=torch.float32, device=device)

            for i_t in range(chunks_in_seq):
                chunk_start = chunk_offset_in_seq + i_t * BT
                chunk_end = min(chunk_start + BT, chunk_offset_in_seq + T_n)
                BT_eff = chunk_end - chunk_start
                if BT_eff <= 0:
                    continue

                h_out[h_batch_idx, boh + i_t, h_idx] = state_kv.transpose(-1, -2).to(k.dtype)

                w_block = w[in_batch_idx, chunk_start:chunk_end, h_idx].float()
                u_block = u[in_batch_idx, chunk_start:chunk_end, h_idx].float()
                v_corr = u_block - torch.einsum("tk,kv->tv", w_block, state_kv)

                if save_new_value:
                    v_new[in_batch_idx, chunk_start:chunk_end, h_idx] = v_corr.to(u.dtype)

                if g is not None:
                    g_block = g[in_batch_idx, chunk_start:chunk_end, h_idx].float()
                    g_last = g_block[-1]
                    rewind = torch.exp(g_last - g_block)
                    bulk_decay = torch.exp(g_last)
                    v_corr = v_corr * rewind[:, None]
                    state_kv = state_kv * bulk_decay

                k_block = k[in_batch_idx, chunk_start:chunk_end, kh].float()
                state_kv = state_kv + torch.einsum("tk,tv->kv", k_block, v_corr)

            if output_final_state:
                final_state[n, h_idx] = state_kv.transpose(-1, -2)

    return h_out, v_new, final_state


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
    dtype: torch.dtype,
    device: torch.device,
    use_g: bool = True,
    use_initial_state: bool = False,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(B * 1000 + T + H + K_dim + seed)
    k = torch.randn(B, T, Hg, K_dim, dtype=dtype, device=device, generator=g) * 0.3
    w = torch.randn(B, T, H, K_dim, dtype=dtype, device=device, generator=g) * 0.3
    u = torch.randn(B, T, H, V_dim, dtype=dtype, device=device, generator=g) * 0.3
    g_tensor = (
        torch.randn(B, T, H, dtype=torch.float32, device=device, generator=g) * 0.05
        if use_g else None
    )
    initial_state = None
    if use_initial_state:
        initial_state = (
            torch.randn(B, H, V_dim, K_dim, dtype=torch.float32, device=device, generator=g) * 0.3
        )
    return {
        "k": k, "w": w, "u": u, "g": g_tensor,
        "initial_state": initial_state,
    }


def _cu_seqlens(seqlens: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [0] + [sum(seqlens[: i + 1]) for i in range(len(seqlens))],
        dtype=torch.int32,
        device=device,
    )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GDN_ChunkGatedDeltaRuleFwdH(Zentorch_TestCase):

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
            B=1, T=T_total, Hg=Hg, H=H, K_dim=K_dim, V_dim=V_dim,
            dtype=dtype, device=cpu, use_g=True, use_initial_state=False,
        )
        cu = _cu_seqlens(seqlens, cpu)
        chunk_offsets = prepare_chunk_offsets(cu, BT)

        NT_total = int(chunk_offsets[-1].item())
        h_cpp, v_new_cpp, fs_cpp = (
            torch.ops.zentorch.gdn_chunk_gated_delta_rule_fwd_h(
                inputs["k"], inputs["w"], inputs["u"], inputs["g"],
                inputs["initial_state"],
                True, BT, True,
                cu, chunk_offsets, NT_total,
            )
        )
        h_oracle, v_new_oracle, fs_oracle = chunk_gated_delta_rule_fwd_h_oracle(
            k=inputs["k"], w=inputs["w"], u=inputs["u"], g=inputs["g"],
            initial_state=inputs["initial_state"],
            output_final_state=True, chunk_size=BT, save_new_value=True,
            cu_seqlens=cu,
        )
        atol, rtol = default_tolerance(dtype)
        self.assertEqual(h_cpp, h_oracle, atol=atol, rtol=rtol, msg="h mismatch")
        self.assertEqual(v_new_cpp, v_new_oracle, atol=atol, rtol=rtol,
                         msg="v_new mismatch")
        self.assertEqual(fs_cpp, fs_oracle, atol=atol, rtol=rtol,
                         msg="final_state mismatch")

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
            K_dim=s.head_k_dim, V_dim=s.head_v_dim,
            dtype=dtype, device=cpu, use_g=True, use_initial_state=False,
        )
        cu = _cu_seqlens(seqlens, cpu)
        chunk_offsets = prepare_chunk_offsets(cu, BT)

        NT_total = int(chunk_offsets[-1].item())
        h_cpp, v_new_cpp, fs_cpp = (
            torch.ops.zentorch.gdn_chunk_gated_delta_rule_fwd_h(
                inputs["k"], inputs["w"], inputs["u"], inputs["g"],
                inputs["initial_state"],
                True, BT, True,
                cu, chunk_offsets, NT_total,
            )
        )
        h_oracle, v_new_oracle, fs_oracle = chunk_gated_delta_rule_fwd_h_oracle(
            k=inputs["k"], w=inputs["w"], u=inputs["u"], g=inputs["g"],
            initial_state=inputs["initial_state"],
            output_final_state=True, chunk_size=BT, save_new_value=True,
            cu_seqlens=cu,
        )
        atol, rtol = default_tolerance(dtype)
        self.assertEqual(h_cpp, h_oracle, atol=atol, rtol=rtol, msg="h mismatch")
        self.assertEqual(v_new_cpp, v_new_oracle, atol=atol, rtol=rtol,
                         msg="v_new mismatch")
        self.assertEqual(fs_cpp, fs_oracle, atol=atol, rtol=rtol,
                         msg="final_state mismatch")


if __name__ == "__main__":
    run_tests()
