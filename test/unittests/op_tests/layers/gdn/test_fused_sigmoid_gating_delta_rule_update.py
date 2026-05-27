# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for ``fused_sigmoid_gating_delta_rule_update``.

Oracle drives HF's ``torch_recurrent_gated_delta_rule`` one token at a time
per sequence to support per-token cache slots (spec-decode).
"""

from __future__ import annotations

import sys
import unittest
from itertools import product
from pathlib import Path

import torch
from parameterized import parameterized
from transformers.models.qwen3_next.modeling_qwen3_next import (
    torch_recurrent_gated_delta_rule as _hf_torch_recurrent_gated_delta_rule,
)

from test.unittests.op_tests.layers.gdn.helpers.shapes import Qwen35_4B_GDN

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from unittest_utils import (  # noqa: E402
    Zentorch_TestCase,
    default_tolerance,
    has_zentorch,
    run_tests,
)


DTYPES = [torch.float32, torch.bfloat16]
STATE_DTYPES = [torch.float32, torch.bfloat16]
GQA_SHAPES = [
    ("gqa2", 2, 4, 8, 8),
    ("gqa1", 4, 4, 8, 8),
    ("gqa4", 2, 8, 16, 32),
]
SEQLENS_PLAIN = [
    [1],
    [1, 1, 1],
    [3],
]
USE_QK_L2NORM = [True, False]
SPEC_DECODE_CASES = [
    ("seq1_acc3", [3], [3], [[1, 2, 3]]),
    ("seq1_acc2", [4], [2], [[1, 2, 3, 4]]),
    ("seq2", [2, 3], [1, 3], [[1, 2, 0, 0], [3, 4, 5, 0]]),
]
QWEN_BATCHES = [1, 8]

_NULL_BLOCK_ID: int = 0


def _stable_softplus(x: torch.Tensor, beta_temp: float, threshold: float) -> torch.Tensor:
    bx = beta_temp * x
    stable = bx.clamp(min=0.0) + torch.log1p(torch.exp(-bx.abs()))
    stable = stable / beta_temp
    return torch.where(bx <= threshold, stable, x)


def _lookup_state_index(
    ssm_state_indices: torch.Tensor, *, n: int, t: int,
) -> int:
    if ssm_state_indices.dim() == 1:
        return int(ssm_state_indices[n].item())
    return int(ssm_state_indices[n, t].item())


def fused_sigmoid_gating_delta_rule_update_oracle(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    beta: float = 1.0,
    threshold: float = 20.0,
    scale: float | None = None,
    initial_state: torch.Tensor,
    inplace_final_state: bool = True,
    cu_seqlens: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    num_accepted_tokens: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    is_kda: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent oracle for ``fused_sigmoid_gating_delta_rule_update``."""
    assert not is_kda, "is_kda not supported"
    assert inplace_final_state, "inplace_final_state=False not supported"

    B, T, H, K = q.shape
    _, _, HV, V = v.shape
    r = HV // H
    if scale is None:
        scale = K ** -0.5

    out_dtype = q.dtype
    state_dtype = initial_state.dtype
    device = q.device

    a_f = a.float()
    b_f = b.float()
    A_log_f = A_log.float()
    dt_bias_f = dt_bias.float()

    sp = _stable_softplus(a_f + dt_bias_f, beta_temp=beta, threshold=threshold)
    g = -torch.exp(A_log_f) * sp
    beta_out = torch.sigmoid(b_f)

    o = torch.empty(B, T, HV, V, dtype=out_dtype, device=device)

    N = cu_seqlens.numel() - 1
    for n in range(N):
        bos = int(cu_seqlens[n].item())
        eos = int(cu_seqlens[n + 1].item())
        T_n = eos - bos
        if T_n <= 0:
            continue

        i_t0 = (
            int(num_accepted_tokens[n].item()) - 1
            if num_accepted_tokens is not None else 0
        )
        init_idx = _lookup_state_index(ssm_state_indices, n=n, t=i_t0)
        if init_idx <= _NULL_BLOCK_ID:
            # Null-slot contract: zero-fill the skipped sequence's range
            # (mirrors C++ op). Without this the torch.empty(...) allocation
            # above would leak uninitialised values into the comparison.
            o[0, bos : bos + T_n] = 0
            continue

        h_kv = (
            initial_state[init_idx]
            .clone()
            .float()
            .transpose(-1, -2)
            .unsqueeze(0)
            .contiguous()
        )

        for t in range(T_n):
            pos = bos + t
            q_slice = q[:, pos : pos + 1].repeat_interleave(r, dim=2)
            k_slice = k[:, pos : pos + 1].repeat_interleave(r, dim=2)
            v_slice = v[:, pos : pos + 1]
            g_slice = g[:, pos : pos + 1]
            beta_slice = beta_out[:, pos : pos + 1]

            out_slice, h_kv = _hf_torch_recurrent_gated_delta_rule(
                query=q_slice,
                key=k_slice,
                value=v_slice,
                g=g_slice,
                beta=beta_slice,
                initial_state=h_kv,
                output_final_state=True,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )
            o[0, pos] = out_slice.squeeze(0).squeeze(0).to(out_dtype)

            final_idx = _lookup_state_index(ssm_state_indices, n=n, t=t)
            if final_idx > _NULL_BLOCK_ID:
                initial_state[final_idx] = (
                    h_kv.squeeze(0).transpose(-1, -2).to(state_dtype)
                )

    return o, initial_state


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _seqlens_name(seqlens: list[int]) -> str:
    return "x".join(map(str, seqlens))


def _name_plain(method, _idx, params):
    dtype, state_dtype, gqa, seqlens, use_qk_l2norm = params.args
    l2_tag = "l2norm" if use_qk_l2norm else "no_l2norm"
    return (
        f"{method.__name__}_{_dtype_name(dtype)}_"
        f"state_{_dtype_name(state_dtype)}_{gqa[0]}_"
        f"{_seqlens_name(seqlens)}_{l2_tag}"
    )


def _name_spec(method, _idx, params):
    dtype, case = params.args
    return f"{method.__name__}_{_dtype_name(dtype)}_{case[0]}"


def _name_dtype_batch(method, _idx, params):
    dtype, batch = params.args
    return f"{method.__name__}_{_dtype_name(dtype)}_b{batch}"


def _build_inputs(
    *,
    seqlens: list[int],
    H: int,
    HV: int,
    K: int,
    V: int,
    num_cache_lines: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    device: torch.device,
    cache_indices_1d: list[int] | None = None,
    cache_indices_2d: list[list[int]] | None = None,
    num_accepted: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    assert (cache_indices_1d is None) != (cache_indices_2d is None), \
        "Provide exactly one of cache_indices_1d or cache_indices_2d"

    g = torch.Generator(device=device).manual_seed(sum(seqlens) * 1000 + HV * V)

    T_total = sum(seqlens)
    a = torch.randn(1, T_total, HV, dtype=dtype, device=device, generator=g) * 0.5
    b = torch.randn(1, T_total, HV, dtype=dtype, device=device, generator=g) * 0.5
    A_log = torch.randn(HV, dtype=torch.float32, device=device, generator=g) * 0.1
    dt_bias = torch.randn(HV, dtype=torch.float32, device=device, generator=g) * 0.1
    q = torch.randn(1, T_total, H, K, dtype=dtype, device=device, generator=g) * 0.5
    k = torch.randn(1, T_total, H, K, dtype=dtype, device=device, generator=g) * 0.5
    v = torch.randn(1, T_total, HV, V, dtype=dtype, device=device, generator=g) * 0.5
    initial_state = (
        torch.randn(num_cache_lines, HV, V, K,
                    dtype=state_dtype, device=device, generator=g) * 0.3
    )

    qsl = [0]
    for sl in seqlens:
        qsl.append(qsl[-1] + sl)
    cu_seqlens = torch.tensor(qsl, dtype=torch.int32, device=device)

    if cache_indices_1d is not None:
        ssm_state_indices = torch.tensor(
            cache_indices_1d, dtype=torch.int32, device=device,
        )
    else:
        ssm_state_indices = torch.tensor(
            cache_indices_2d, dtype=torch.int32, device=device,
        )

    num_accepted_tokens = (
        torch.tensor(num_accepted, dtype=torch.int32, device=device)
        if num_accepted is not None else None
    )

    return {
        "A_log": A_log,
        "a": a, "b": b,
        "dt_bias": dt_bias,
        "q": q, "k": k, "v": v,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "ssm_state_indices": ssm_state_indices,
        "num_accepted_tokens": num_accepted_tokens,
    }


def _run_cpp_and_oracle(
    inputs: dict[str, torch.Tensor],
    *,
    use_qk_l2norm: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    is_for_cpp = inputs["initial_state"].clone()
    is_for_oracle = inputs["initial_state"].clone()

    o_cpp = torch.ops.zentorch.gdn_fused_sigmoid_gating_delta_rule_update(
        inputs["A_log"], inputs["a"], inputs["b"], inputs["dt_bias"],
        inputs["q"], inputs["k"], inputs["v"],
        1.0, 20.0, inputs["q"].size(-1) ** -0.5,
        is_for_cpp,
        inputs["cu_seqlens"], inputs["ssm_state_indices"],
        inputs["num_accepted_tokens"],
        use_qk_l2norm,
    )
    o_oracle, _ = fused_sigmoid_gating_delta_rule_update_oracle(
        A_log=inputs["A_log"],
        a=inputs["a"], b=inputs["b"],
        dt_bias=inputs["dt_bias"],
        q=inputs["q"], k=inputs["k"], v=inputs["v"],
        initial_state=is_for_oracle,
        inplace_final_state=True,
        cu_seqlens=inputs["cu_seqlens"],
        ssm_state_indices=inputs["ssm_state_indices"],
        num_accepted_tokens=inputs["num_accepted_tokens"],
        use_qk_l2norm_in_kernel=use_qk_l2norm,
    )
    return o_cpp, is_for_cpp, o_oracle, is_for_oracle


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GDN_FusedSigmoidGatingDeltaRuleUpdate(Zentorch_TestCase):

    def _check_outputs(
        self,
        o_cpp: torch.Tensor,
        is_cpp: torch.Tensor,
        o_oracle: torch.Tensor,
        is_oracle: torch.Tensor,
        *,
        cache_indices: torch.Tensor,
        out_dtype: torch.dtype,
        state_dtype: torch.dtype,
    ) -> None:
        atol, rtol = default_tolerance(out_dtype, state_dtype)
        self.assertEqual(o_cpp, o_oracle, atol=atol, rtol=rtol, msg="output mismatch")
        flat_ci = cache_indices.flatten().to(torch.long)
        used = flat_ci[flat_ci > 0].unique()
        if used.numel() > 0:
            self.assertEqual(
                is_cpp[used], is_oracle[used],
                atol=atol, rtol=rtol, msg="cache state mismatch",
            )

    @parameterized.expand(
        list(product(DTYPES, STATE_DTYPES, GQA_SHAPES, SEQLENS_PLAIN, USE_QK_L2NORM)),
        name_func=_name_plain,
    )
    def test_cpp_plain_decode_matches_oracle(
        self,
        dtype: torch.dtype,
        state_dtype: torch.dtype,
        gqa: tuple,
        seqlens: list[int],
        use_qk_l2norm: bool,
    ) -> None:
        cpu = torch.device("cpu")
        _, H, HV, K, V = gqa
        n_seqs = len(seqlens)
        inputs = _build_inputs(
            seqlens=seqlens, H=H, HV=HV, K=K, V=V,
            num_cache_lines=n_seqs + 2,
            dtype=dtype, state_dtype=state_dtype, device=cpu,
            cache_indices_1d=[i + 1 for i in range(n_seqs)],
        )

        o_cpp, is_cpp, o_oracle, is_oracle = _run_cpp_and_oracle(
            inputs, use_qk_l2norm=use_qk_l2norm,
        )

        self.assertEqual(o_cpp.shape, (1, sum(seqlens), HV, V))
        self.assertEqual(o_cpp.dtype, dtype)
        self._check_outputs(
            o_cpp, is_cpp, o_oracle, is_oracle,
            cache_indices=inputs["ssm_state_indices"],
            out_dtype=dtype, state_dtype=state_dtype,
        )

    @parameterized.expand(
        list(product(DTYPES, SPEC_DECODE_CASES)),
        name_func=_name_spec,
    )
    def test_cpp_spec_decode_matches_oracle(
        self,
        dtype: torch.dtype,
        case: tuple,
    ) -> None:
        cpu = torch.device("cpu")
        _, seqlens, num_accepted, cache_indices = case
        H, HV, K, V = 2, 4, 8, 16
        inputs = _build_inputs(
            seqlens=seqlens, H=H, HV=HV, K=K, V=V,
            num_cache_lines=8,
            dtype=dtype, state_dtype=torch.float32, device=cpu,
            cache_indices_2d=cache_indices, num_accepted=num_accepted,
        )

        o_cpp, is_cpp, o_oracle, is_oracle = _run_cpp_and_oracle(
            inputs, use_qk_l2norm=True,
        )

        self._check_outputs(
            o_cpp, is_cpp, o_oracle, is_oracle,
            cache_indices=inputs["ssm_state_indices"],
            out_dtype=dtype, state_dtype=torch.float32,
        )

    @parameterized.expand(
        list(product([torch.bfloat16], QWEN_BATCHES)),
        name_func=_name_dtype_batch,
    )
    def test_qwen35_4b_decode_shape(
        self,
        dtype: torch.dtype,
        batch: int,
    ) -> None:
        s = Qwen35_4B_GDN
        cpu = torch.device("cpu")
        inputs = _build_inputs(
            seqlens=[1] * batch,
            H=s.num_k_heads, HV=s.num_v_heads,
            K=s.head_k_dim, V=s.head_v_dim,
            num_cache_lines=batch + 4,
            dtype=dtype, state_dtype=torch.float32, device=cpu,
            cache_indices_1d=[i + 1 for i in range(batch)],
        )
        o_cpp, is_cpp, o_oracle, is_oracle = _run_cpp_and_oracle(
            inputs, use_qk_l2norm=True,
        )
        self._check_outputs(
            o_cpp, is_cpp, o_oracle, is_oracle,
            cache_indices=inputs["ssm_state_indices"],
            out_dtype=dtype, state_dtype=torch.float32,
        )


if __name__ == "__main__":
    run_tests()
