# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for ``fused_recurrent_gated_delta_rule_packed_decode``."""

from __future__ import annotations

import sys
import unittest
from itertools import product
from pathlib import Path

import torch
import torch.nn.functional as F
from parameterized import parameterized

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
SHAPES_BHVKV = [
    ("1x1x1x4x4", 1, 1, 1, 4, 4),
    ("2x1x2x4x8", 2, 1, 2, 4, 8),
    ("4x2x4x8x16", 4, 2, 4, 8, 16),
    ("1x2x8x16x32", 1, 2, 8, 16, 32),
]
USE_QK_L2NORM = [True, False]
PARAM_DTYPE_CASES = [
    (torch.float32, torch.bfloat16),
    (torch.bfloat16, torch.bfloat16),
    (torch.bfloat16, torch.float32),
]
LAYOUT_CASES = [
    ("bf16_x_bf16state", torch.bfloat16, torch.bfloat16),
    ("fp32_x_fp32state", torch.float32, torch.float32),
]
QWEN_BATCHES = [1, 8]

_NULL_BLOCK_ID: int = 0
_SOFTPLUS_THRESHOLD: float = 20.0
_L2NORM_EPS: float = 1e-6


def fused_recurrent_gated_delta_rule_packed_decode_oracle(
    mixed_qkv: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    out: torch.Tensor,
    ssm_state_indices: torch.Tensor,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent oracle for ``fused_recurrent_gated_delta_rule_packed_decode``."""
    B = mixed_qkv.shape[0]
    HV = initial_state.shape[1]
    V = initial_state.shape[2]
    K = initial_state.shape[3]

    qkv_dim = mixed_qkv.shape[1]
    q_dim = (qkv_dim - HV * V) // 2
    H = q_dim // K
    r = HV // H

    out_dtype = out.dtype
    state_dtype = initial_state.dtype

    q_flat, k_flat, v_flat = torch.split(
        mixed_qkv.float(), [H * K, H * K, HV * V], dim=-1,
    )
    q_all = q_flat.view(B, H, K)
    k_all = k_flat.view(B, H, K)
    v_all = v_flat.view(B, HV, V)

    if use_qk_l2norm_in_kernel:
        q_all = F.normalize(q_all, p=2, dim=-1, eps=_L2NORM_EPS)
        k_all = F.normalize(k_all, p=2, dim=-1, eps=_L2NORM_EPS)
    q_all = q_all * scale

    x = a.float() + dt_bias.float().unsqueeze(0)
    softplus_x = F.softplus(x, beta=1.0, threshold=_SOFTPLUS_THRESHOLD)
    g_all = -torch.exp(A_log.float()).unsqueeze(0) * softplus_x
    beta_all = torch.sigmoid(b.float())

    for b_idx in range(B):
        slot = int(ssm_state_indices[b_idx].item())
        if slot <= _NULL_BLOCK_ID:
            out[b_idx, 0, :, :] = 0
            continue

        state_KV = initial_state[slot].float().transpose(-1, -2).contiguous()

        for hv in range(HV):
            i_h = hv // r
            q_h = q_all[b_idx, i_h]
            k_h = k_all[b_idx, i_h]
            v_h = v_all[b_idx, hv]
            g_h = g_all[b_idx, hv]
            beta_h = beta_all[b_idx, hv]

            state = state_KV[hv]
            state = state * torch.exp(g_h)

            v_correction = torch.einsum("kv,k->v", state, k_h)
            v_corrected = (v_h - v_correction) * beta_h

            state = state + torch.einsum("k,v->kv", k_h, v_corrected)

            o_h = torch.einsum("kv,k->v", state, q_h)
            out[b_idx, 0, hv, :] = o_h.to(out_dtype)
            state_KV[hv] = state

        initial_state[slot] = state_KV.transpose(-1, -2).contiguous().to(state_dtype)

    return out, initial_state


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _name_oracle(method, _idx, params):
    dtype, state_dtype, shape, use_qk_l2norm = params.args
    l2_tag = "l2norm" if use_qk_l2norm else "no_l2norm"
    return (
        f"{method.__name__}_{_dtype_name(dtype)}_"
        f"state_{_dtype_name(state_dtype)}_{shape[0]}_{l2_tag}"
    )


def _name_param_dtype(method, _idx, params):
    A_log_dtype, dt_bias_dtype = params.args
    return (
        f"{method.__name__}_"
        f"Alog_{_dtype_name(A_log_dtype)}_dt_{_dtype_name(dt_bias_dtype)}"
    )


def _name_layout(method, _idx, params):
    case = params.args[0]
    return f"{method.__name__}_{case[0]}"


def _name_dtype_batch(method, _idx, params):
    dtype, batch = params.args
    return f"{method.__name__}_{_dtype_name(dtype)}_b{batch}"


def _build_inputs(
    *,
    B: int,
    H: int, HV: int, K: int, V: int,
    num_cache_lines: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    device: torch.device,
    cache_indices: list[int] | None = None,
    seed: int = 0,
    A_log_dtype: torch.dtype = torch.float32,
    dt_bias_dtype: torch.dtype = torch.float32,
    state_layout: str = "contiguous",
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(B * 1000 + HV * V + K + seed)

    qkv_dim = 2 * H * K + HV * V
    mixed_qkv = torch.randn(B, qkv_dim, dtype=dtype, device=device, generator=g) * 0.3
    a = torch.randn(B, HV, dtype=dtype, device=device, generator=g) * 0.5
    b = torch.randn(B, HV, dtype=dtype, device=device, generator=g) * 0.5
    A_log = (torch.randn(HV, dtype=torch.float32, device=device, generator=g)
             * 0.1).to(A_log_dtype)
    dt_bias = (torch.randn(HV, dtype=torch.float32, device=device, generator=g)
               * 0.1).to(dt_bias_dtype)
    initial_state = (
        torch.randn(num_cache_lines, HV, V, K,
                    dtype=state_dtype, device=device, generator=g) * 0.3
    )
    if state_layout == "vllm_swapped":
        initial_state = initial_state.transpose(0, 1).contiguous().transpose(0, 1)
        assert not initial_state.is_contiguous()
        assert initial_state.stride(3) == 1
        assert initial_state.stride(2) == K
    elif state_layout != "contiguous":
        raise ValueError(f"Unknown state_layout: {state_layout!r}")
    if cache_indices is None:
        cache_indices = [i + 1 for i in range(B)]
    ssm_state_indices = torch.tensor(cache_indices, dtype=torch.int32, device=device)
    out = torch.zeros(B, 1, HV, V, dtype=dtype, device=device)

    return {
        "mixed_qkv": mixed_qkv,
        "a": a, "b": b,
        "A_log": A_log, "dt_bias": dt_bias,
        "initial_state": initial_state,
        "out": out,
        "ssm_state_indices": ssm_state_indices,
    }


def _run_cpp_and_oracle(
    inputs: dict[str, torch.Tensor],
    *,
    use_qk_l2norm: bool,
    scale: float,
) -> tuple[
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, torch.Tensor],
]:
    is_cpp = inputs["initial_state"].clone()
    is_or = inputs["initial_state"].clone()
    out_cpp = inputs["out"].clone()
    out_or = inputs["out"].clone()

    common = {
        "mixed_qkv": inputs["mixed_qkv"],
        "a": inputs["a"],
        "b": inputs["b"],
        "A_log": inputs["A_log"],
        "dt_bias": inputs["dt_bias"],
        "scale": scale,
        "ssm_state_indices": inputs["ssm_state_indices"],
        "use_qk_l2norm_in_kernel": use_qk_l2norm,
    }
    torch.ops.zentorch.gdn_fused_recurrent_gated_delta_rule_packed_decode(
        common["mixed_qkv"],
        common["a"], common["b"],
        common["A_log"], common["dt_bias"],
        common["scale"],
        is_cpp, out_cpp,
        common["ssm_state_indices"],
        common["use_qk_l2norm_in_kernel"],
    )
    fused_recurrent_gated_delta_rule_packed_decode_oracle(
        **common, initial_state=is_or, out=out_or,
    )
    return (out_cpp, is_cpp), (out_or, is_or)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GDN_FusedRecurrentGatedDeltaRulePackedDecode(Zentorch_TestCase):

    @parameterized.expand(
        list(product(DTYPES, STATE_DTYPES, SHAPES_BHVKV, USE_QK_L2NORM)),
        name_func=_name_oracle,
    )
    def test_cpp_matches_oracle(
        self,
        dtype: torch.dtype,
        state_dtype: torch.dtype,
        shape: tuple,
        use_qk_l2norm: bool,
    ) -> None:
        cpu = torch.device("cpu")
        _, B, H, HV, K, V = shape
        inputs = _build_inputs(
            B=B, H=H, HV=HV, K=K, V=V,
            num_cache_lines=B + 2,
            dtype=dtype, state_dtype=state_dtype, device=cpu,
        )
        scale = K ** -0.5

        (out_cpp, is_cpp), (out_or, is_or) = _run_cpp_and_oracle(
            inputs, use_qk_l2norm=use_qk_l2norm, scale=scale,
        )

        self.assertEqual(out_cpp.shape, (B, 1, HV, V))
        self.assertEqual(out_cpp.dtype, dtype)

        atol, rtol = default_tolerance(dtype, state_dtype)
        self.assertEqual(out_cpp, out_or, atol=atol, rtol=rtol, msg="out mismatch")
        used_slots = inputs["ssm_state_indices"][inputs["ssm_state_indices"] > 0].long()
        if used_slots.numel() > 0:
            self.assertEqual(
                is_cpp[used_slots], is_or[used_slots],
                atol=atol, rtol=rtol, msg="initial_state mismatch",
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
            B=batch,
            H=s.num_k_heads, HV=s.num_v_heads,
            K=s.head_k_dim, V=s.head_v_dim,
            num_cache_lines=batch + 4,
            dtype=dtype, state_dtype=dtype, device=cpu,
            A_log_dtype=torch.float32,
            dt_bias_dtype=dtype,
        )
        scale = s.head_k_dim ** -0.5

        (out_cpp, is_cpp), (out_or, is_or) = _run_cpp_and_oracle(
            inputs, use_qk_l2norm=True, scale=scale,
        )
        atol, rtol = default_tolerance(dtype, dtype)
        self.assertEqual(out_cpp, out_or, atol=atol, rtol=rtol)
        used_slots = inputs["ssm_state_indices"].long()
        self.assertEqual(is_cpp[used_slots], is_or[used_slots], atol=atol, rtol=rtol)

    @parameterized.expand(
        PARAM_DTYPE_CASES,
        name_func=_name_param_dtype,
    )
    def test_param_dtype_variations(
        self,
        A_log_dtype: torch.dtype,
        dt_bias_dtype: torch.dtype,
    ) -> None:
        cpu = torch.device("cpu")
        B, H, HV, K, V = 2, 1, 2, 4, 8
        inputs = _build_inputs(
            B=B, H=H, HV=HV, K=K, V=V,
            num_cache_lines=B + 2,
            dtype=torch.bfloat16,
            state_dtype=torch.bfloat16,
            device=cpu,
            A_log_dtype=A_log_dtype,
            dt_bias_dtype=dt_bias_dtype,
        )
        scale = K ** -0.5

        (out_cpp, is_cpp), (out_or, is_or) = _run_cpp_and_oracle(
            inputs, use_qk_l2norm=True, scale=scale,
        )
        atol, rtol = default_tolerance(torch.bfloat16, torch.bfloat16)
        self.assertEqual(out_cpp, out_or, atol=atol, rtol=rtol, msg="out mismatch")
        used_slots = inputs["ssm_state_indices"][inputs["ssm_state_indices"] > 0].long()
        if used_slots.numel() > 0:
            self.assertEqual(
                is_cpp[used_slots], is_or[used_slots],
                atol=atol, rtol=rtol, msg="initial_state mismatch",
            )

    @parameterized.expand(
        [(case,) for case in LAYOUT_CASES],
        name_func=_name_layout,
    )
    def test_non_contiguous_initial_state(
        self,
        case: tuple,
    ) -> None:
        cpu = torch.device("cpu")
        _, dtype, state_dtype = case
        B, H, HV, K, V = 2, 1, 2, 4, 8
        inputs = _build_inputs(
            B=B, H=H, HV=HV, K=K, V=V,
            num_cache_lines=B + 4,
            dtype=dtype, state_dtype=state_dtype, device=cpu,
            state_layout="vllm_swapped",
        )
        scale = K ** -0.5

        (out_cpp, is_cpp), (out_or, is_or) = _run_cpp_and_oracle(
            inputs, use_qk_l2norm=True, scale=scale,
        )
        atol, rtol = default_tolerance(dtype, state_dtype)
        self.assertEqual(out_cpp, out_or, atol=atol, rtol=rtol, msg="out mismatch")
        used_slots = inputs["ssm_state_indices"][inputs["ssm_state_indices"] > 0].long()
        if used_slots.numel() > 0:
            self.assertEqual(
                is_cpp[used_slots], is_or[used_slots],
                atol=atol, rtol=rtol, msg="initial_state mismatch",
            )

    def test_null_block_id_writes_zero_output(self) -> None:
        cpu = torch.device("cpu")
        H, HV, K, V = 1, 2, 4, 4
        inputs = _build_inputs(
            B=3, H=H, HV=HV, K=K, V=V,
            num_cache_lines=5,
            dtype=torch.float32, state_dtype=torch.float32, device=cpu,
            cache_indices=[1, 0, 2],
        )
        cache_before = inputs["initial_state"].clone()

        torch.ops.zentorch.gdn_fused_recurrent_gated_delta_rule_packed_decode(
            inputs["mixed_qkv"],
            inputs["a"], inputs["b"],
            inputs["A_log"], inputs["dt_bias"],
            K ** -0.5,
            inputs["initial_state"], inputs["out"],
            inputs["ssm_state_indices"],
            False,
        )

        self.assertTrue(torch.all(inputs["out"][1, 0] == 0))
        self.assertFalse(torch.equal(inputs["initial_state"][1], cache_before[1]))
        self.assertFalse(torch.equal(inputs["initial_state"][2], cache_before[2]))
        for slot in (0, 3, 4):
            with self.subTest(slot=slot):
                self.assertTrue(
                    torch.equal(inputs["initial_state"][slot], cache_before[slot]),
                )

    def test_negative_state_index_skips(self) -> None:
        cpu = torch.device("cpu")
        inputs = _build_inputs(
            B=2, H=1, HV=2, K=4, V=4,
            num_cache_lines=4,
            dtype=torch.float32, state_dtype=torch.float32, device=cpu,
            cache_indices=[1, -3],
        )
        cache_before = inputs["initial_state"].clone()
        torch.ops.zentorch.gdn_fused_recurrent_gated_delta_rule_packed_decode(
            inputs["mixed_qkv"], inputs["a"], inputs["b"],
            inputs["A_log"], inputs["dt_bias"],
            4 ** -0.5,
            inputs["initial_state"], inputs["out"],
            inputs["ssm_state_indices"],
            False,
        )
        self.assertTrue(torch.all(inputs["out"][1, 0] == 0))
        self.assertTrue(torch.equal(inputs["initial_state"][0], cache_before[0]))
        self.assertFalse(torch.equal(inputs["initial_state"][1], cache_before[1]))

    def test_state_layout_round_trip(self) -> None:
        cpu = torch.device("cpu")
        inputs = _build_inputs(
            B=2, H=1, HV=2, K=4, V=4,
            num_cache_lines=8,
            dtype=torch.float32, state_dtype=torch.float32, device=cpu,
            cache_indices=[3, 5],
        )
        cache_before = inputs["initial_state"].clone()
        torch.ops.zentorch.gdn_fused_recurrent_gated_delta_rule_packed_decode(
            inputs["mixed_qkv"], inputs["a"], inputs["b"],
            inputs["A_log"], inputs["dt_bias"],
            4 ** -0.5,
            inputs["initial_state"], inputs["out"],
            inputs["ssm_state_indices"],
            False,
        )
        self.assertFalse(torch.equal(inputs["initial_state"][3], cache_before[3]))
        self.assertFalse(torch.equal(inputs["initial_state"][5], cache_before[5]))
        for slot in (0, 1, 2, 4, 6, 7):
            with self.subTest(slot=slot):
                self.assertTrue(
                    torch.equal(inputs["initial_state"][slot], cache_before[slot]),
                )


if __name__ == "__main__":
    run_tests()
