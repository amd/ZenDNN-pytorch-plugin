# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for ``chunk_gated_delta_rule_fwd``.

Oracle wraps HuggingFace's ``torch_chunk_gated_delta_rule`` and bridges
GQA expansion, ``(V, K)`` vs ``(K, V)`` state layout, and varlen batching.
"""

from __future__ import annotations

import sys
import unittest
from itertools import product
from pathlib import Path

import torch
from parameterized import parameterized
from transformers.models.qwen3_next.modeling_qwen3_next import (
    torch_chunk_gated_delta_rule as _hf_torch_chunk_gated_delta_rule,
)

from test.unittests.op_tests.layers.gdn.helpers.shapes import Qwen35_4B_GDN
from test.unittests.op_tests.layers.gdn.helpers.varlen import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from unittest_utils import (  # noqa: E402
    Zentorch_TestCase,
    default_tolerance,
    has_zentorch,
    run_tests,
)


DTYPES = [torch.float32, torch.bfloat16]
SEQLENS_VARLEN = [
    [64],
    [128],
    [50],
    [80, 30, 64],
]
SEQLENS_QWEN = [[64], [128, 64]]

_FLA_CHUNK_SIZE = 64
_L2NORM_EPS = 1e-6


def _gqa_expand(x: torch.Tensor, ratio: int) -> torch.Tensor:
    if ratio == 1:
        return x
    return x.repeat_interleave(ratio, dim=-2)


def _vK_to_KV(state_VK: torch.Tensor) -> torch.Tensor:
    return state_VK.transpose(-1, -2).contiguous()


def _KV_to_VK(state_KV: torch.Tensor) -> torch.Tensor:
    return state_KV.transpose(-1, -2).contiguous()


def _call_hf_one_sequence(
    *,
    q_seq: torch.Tensor,
    k_seq: torch.Tensor,
    v_seq: torch.Tensor,
    g_seq: torch.Tensor,
    beta_seq: torch.Tensor,
    initial_state_KV: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return _hf_torch_chunk_gated_delta_rule(
        query=q_seq,
        key=k_seq,
        value=v_seq,
        g=g_seq,
        beta=beta_seq,
        chunk_size=chunk_size,
        initial_state=initial_state_KV,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )


def chunk_gated_delta_rule_oracle(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Independent oracle for ``chunk_gated_delta_rule_fwd``."""
    B, T, Hg, K_dim = q.shape
    _, _, H, V_dim = v.shape
    r = H // Hg

    # HF applies q * (1/sqrt(K)) internally; absorb the user `scale` ratio into q.
    scale_correction = 1.0 if scale is None else scale / (K_dim ** -0.5)

    q_e = _gqa_expand(q, r)
    k_e = _gqa_expand(k, r)
    if scale_correction != 1.0:
        q_e = q_e * scale_correction

    if cu_seqlens is None:
        init_KV = _vK_to_KV(initial_state.float()) if initial_state is not None else None
        o_hf, last_state_KV = _call_hf_one_sequence(
            q_seq=q_e, k_seq=k_e, v_seq=v, g_seq=g, beta_seq=beta,
            initial_state_KV=init_KV,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            chunk_size=_FLA_CHUNK_SIZE,
        )
        final_state = (
            _KV_to_VK(last_state_KV.float())
            if output_final_state and last_state_KV is not None else None
        )
        return o_hf.to(q.dtype), final_state

    N = cu_seqlens.numel() - 1
    out_dtype = q.dtype
    o_full = torch.zeros(1, T, H, V_dim, dtype=out_dtype, device=q.device)
    final_state = (
        torch.zeros(N, H, V_dim, K_dim, dtype=torch.float32, device=q.device)
        if output_final_state else None
    )

    for n in range(N):
        bos = int(cu_seqlens[n].item())
        eos = int(cu_seqlens[n + 1].item())
        T_n = eos - bos
        if T_n <= 0:
            continue

        q_seq = q_e[:, bos:eos]
        k_seq = k_e[:, bos:eos]
        v_seq = v[:, bos:eos]
        g_seq = g[:, bos:eos]
        beta_seq = beta[:, bos:eos]

        init_KV_n = (
            _vK_to_KV(initial_state[n : n + 1].float())
            if initial_state is not None else None
        )

        o_seq, last_state_KV = _call_hf_one_sequence(
            q_seq=q_seq, k_seq=k_seq, v_seq=v_seq, g_seq=g_seq, beta_seq=beta_seq,
            initial_state_KV=init_KV_n,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            chunk_size=_FLA_CHUNK_SIZE,
        )

        o_full[:, bos:eos] = o_seq.to(out_dtype)
        if output_final_state and last_state_KV is not None:
            final_state[n] = _KV_to_VK(last_state_KV.squeeze(0).float())

    return o_full, final_state


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
    use_initial_state: bool = False,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(B * 1000 + T + H + K_dim + seed)
    q = torch.randn(B, T, Hg, K_dim, dtype=dtype, device=device, generator=g) * 0.3
    k = torch.randn(B, T, Hg, K_dim, dtype=dtype, device=device, generator=g) * 0.3
    v = torch.randn(B, T, H, V_dim, dtype=dtype, device=device, generator=g) * 0.3
    g_tensor = torch.randn(B, T, H, dtype=torch.float32, device=device, generator=g) * 0.05
    beta = torch.randn(B, T, H, dtype=dtype, device=device, generator=g) * 0.2 + 0.5
    initial_state = None
    if use_initial_state:
        initial_state = (
            torch.randn(B, H, V_dim, K_dim, dtype=torch.float32, device=device, generator=g) * 0.3
        )
    return {
        "q": q, "k": k, "v": v, "g": g_tensor, "beta": beta,
        "initial_state": initial_state,
    }


def _compute_tolerances(
    out_dtype: torch.dtype, T_total: int, BT: int = 64,
) -> tuple[float, float]:
    """Tolerance scaled by chunk count: chunked algorithm accumulates fp error per chunk."""
    base_atol, base_rtol = default_tolerance(out_dtype, torch.float32)
    NT = max(1, (T_total + BT - 1) // BT)
    return base_atol * NT, base_rtol * NT


def _cu_seqlens(seqlens: list[int], device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [0] + [sum(seqlens[: i + 1]) for i in range(len(seqlens))],
        dtype=torch.int32,
        device=device,
    )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GDN_ChunkGatedDeltaRuleFwd(Zentorch_TestCase):

    @parameterized.expand(
        list(product(DTYPES, SEQLENS_VARLEN)),
        name_func=_name_dtype_seqlens,
    )
    def test_cpp_matches_hf_oracle_varlen(
        self,
        dtype: torch.dtype,
        seqlens: list[int],
    ) -> None:
        cpu = torch.device("cpu")
        Hg, H, K_dim, V_dim = 2, 4, 8, 16
        T_total = sum(seqlens)
        inputs = _build_inputs(
            B=1, T=T_total, Hg=Hg, H=H, K_dim=K_dim, V_dim=V_dim,
            dtype=dtype, device=cpu,
        )
        cu_seqlens = _cu_seqlens(seqlens, cpu)

        g_rng = torch.Generator(device=cpu).manual_seed(99)
        inputs["initial_state"] = torch.randn(
            len(seqlens), H, V_dim, K_dim,
            dtype=torch.float32, device=cpu, generator=g_rng,
        ) * 0.3

        BT = 64
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        scale = K_dim ** -0.5

        o_cpp, fs_cpp = torch.ops.zentorch.gdn_chunk_gated_delta_rule_fwd(
            inputs["q"], inputs["k"], inputs["v"],
            inputs["g"], inputs["beta"], float(scale),
            inputs["initial_state"], True, BT,
            cu_seqlens, chunk_indices, chunk_offsets,
        )

        o_oracle, fs_oracle = chunk_gated_delta_rule_oracle(
            **inputs,
            cu_seqlens=cu_seqlens,
            output_final_state=True,
        )

        self.assertEqual(o_cpp.shape, (1, T_total, H, V_dim))
        self.assertEqual(o_cpp.dtype, dtype)
        self.assertEqual(fs_cpp.shape, (len(seqlens), H, V_dim, K_dim))
        self.assertEqual(fs_cpp.dtype, torch.float32)

        atol, rtol = _compute_tolerances(dtype, max(seqlens))
        self.assertEqual(o_cpp, o_oracle, atol=atol, rtol=rtol, msg="o mismatch")
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
        T_total = sum(seqlens)
        inputs = _build_inputs(
            B=1, T=T_total,
            Hg=s.num_k_heads, H=s.num_v_heads,
            K_dim=s.head_k_dim, V_dim=s.head_v_dim,
            dtype=dtype, device=cpu,
        )
        cu_seqlens = _cu_seqlens(seqlens, cpu)

        g_rng = torch.Generator(device=cpu).manual_seed(99)
        inputs["initial_state"] = torch.randn(
            len(seqlens), s.num_v_heads, s.head_v_dim, s.head_k_dim,
            dtype=torch.float32, device=cpu, generator=g_rng,
        ) * 0.3

        BT = 64
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        scale = s.head_k_dim ** -0.5

        o_cpp, fs_cpp = torch.ops.zentorch.gdn_chunk_gated_delta_rule_fwd(
            inputs["q"], inputs["k"], inputs["v"],
            inputs["g"], inputs["beta"], float(scale),
            inputs["initial_state"], True, BT,
            cu_seqlens, chunk_indices, chunk_offsets,
        )
        o_oracle, fs_oracle = chunk_gated_delta_rule_oracle(
            **inputs,
            cu_seqlens=cu_seqlens,
            output_final_state=True,
        )

        # Magnitude-adaptive tolerance: at Qwen scale fp error grows with the
        # largest absolute value in the output rather than chunk count.
        rtol = 0.005
        atol_o = max(0.01, rtol * o_oracle.abs().max().item())
        atol_fs = max(0.01, rtol * fs_oracle.abs().max().item())
        self.assertEqual(o_cpp, o_oracle, atol=atol_o, rtol=rtol, msg="o mismatch")
        self.assertEqual(fs_cpp, fs_oracle, atol=atol_fs, rtol=rtol,
                         msg="final_state mismatch")

    def test_cpp_rejects_fp16(self) -> None:
        cpu = torch.device("cpu")
        inputs = _build_inputs(
            B=1, T=64, Hg=1, H=1, K_dim=8, V_dim=8,
            dtype=torch.float16, device=cpu,
        )
        cu_seqlens = torch.tensor([0, 64], dtype=torch.int32, device=cpu)
        chunk_indices = torch.tensor([[0, 0]], dtype=torch.int32, device=cpu)
        chunk_offsets = torch.tensor([0, 1], dtype=torch.int32, device=cpu)

        with self.assertRaisesRegex(RuntimeError, "fp16"):
            torch.ops.zentorch.gdn_chunk_gated_delta_rule_fwd(
                inputs["q"], inputs["k"], inputs["v"],
                inputs["g"], inputs["beta"], 1.0 / (8 ** 0.5),
                None, False, 64,
                cu_seqlens, chunk_indices, chunk_offsets,
            )

    def test_cpp_returns_zero_element_final_state_when_disabled(self) -> None:
        cpu = torch.device("cpu")
        inputs = _build_inputs(
            B=1, T=64, Hg=1, H=1, K_dim=8, V_dim=8,
            dtype=torch.float32, device=cpu,
        )
        cu_seqlens = torch.tensor([0, 64], dtype=torch.int32, device=cpu)
        chunk_indices = torch.tensor([[0, 0]], dtype=torch.int32, device=cpu)
        chunk_offsets = torch.tensor([0, 1], dtype=torch.int32, device=cpu)

        _, fs = torch.ops.zentorch.gdn_chunk_gated_delta_rule_fwd(
            inputs["q"], inputs["k"], inputs["v"],
            inputs["g"], inputs["beta"], 1.0 / (8 ** 0.5),
            None, False, 64,
            cu_seqlens, chunk_indices, chunk_offsets,
        )
        self.assertEqual(fs.numel(), 0)


if __name__ == "__main__":
    run_tests()
