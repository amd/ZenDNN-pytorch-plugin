# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for ``causal_conv1d_fn`` (prefill)."""

from __future__ import annotations

import sys
import unittest
from itertools import product
from pathlib import Path

import torch
from parameterized import parameterized

from test.unittests.op_tests.layers.gdn.helpers.shapes import Qwen35_4B_GDN
from test.unittests.op_tests.layers.gdn.helpers.varlen import PAD_SLOT_ID

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from unittest_utils import (  # noqa: E402
    Zentorch_TestCase,
    default_tolerance,
    has_zentorch,
    run_tests,
)


DTYPES = [torch.float32, torch.bfloat16]
STATE_DTYPES = [torch.float32, torch.bfloat16]
SEQLENS_VARLEN = [
    [7],
    [64],
    [3, 5],
    [10, 1, 17],
    [128, 7],
]
WIDTHS = [4]
ACTIVATIONS = [None, "silu", "swish"]
WITH_BIAS = [True, False]
WITH_INITIAL_STATE = [True, False]

SEQLENS_QWEN = [[64], [257], [128, 64, 7]]


def _resolve_activation(activation: str | bool | None) -> str | None:
    if activation is None:
        return None
    if isinstance(activation, bool):
        return "silu" if activation else None
    if activation in ("silu", "swish"):
        return "silu"
    raise ValueError(f"Unsupported activation: {activation!r}")


def _depthwise_conv1d_manual(
    padded: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Sliding-window depthwise conv via unfold + elementwise multiply + sum."""
    width = weight.shape[1]
    windows = padded.unfold(dimension=-1, size=width, step=1)
    out = (windows * weight.unsqueeze(1)).sum(dim=-1)
    if bias is not None:
        out = out + bias.unsqueeze(-1)
    return out


def causal_conv1d_fn_oracle(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    *,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    activation: str | bool | None = "silu",
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    """Independent oracle for ``causal_conv1d_fn``."""
    act = _resolve_activation(activation)

    dim, cu_seqlen = x.shape
    width = weight.shape[1]
    state_len = width - 1
    batch = query_start_loc.numel() - 1

    original_x_dtype = x.dtype
    compute_dtype = conv_states.dtype
    x_c = x.to(compute_dtype)
    w_c = weight.to(compute_dtype)
    b_c = bias.to(compute_dtype) if bias is not None else None

    # Initialise from x_c so pad-slot sequences preserve input (mirrors the
    # C++ op which uses x.clone(); without this both oracle and op would
    # diverge on uninitialised positions for any skipped non-empty sequence).
    out = x_c.clone()

    for b in range(batch):
        start = int(query_start_loc[b].item())
        end = int(query_start_loc[b + 1].item())
        T_b = end - start
        if T_b <= 0:
            continue
        if cache_indices is not None and int(cache_indices[b].item()) == pad_slot_id:
            continue

        x_b = x_c[:, start:end]

        if has_initial_state is not None and bool(has_initial_state[b].item()):
            slot = int(cache_indices[b].item())
            state = conv_states[slot, :, :state_len].to(compute_dtype)
        else:
            state = torch.zeros(dim, state_len, dtype=compute_dtype, device=x.device)

        padded = torch.cat([state, x_b], dim=-1)
        conv_out = _depthwise_conv1d_manual(padded, w_c, b_c)

        if act == "silu":
            conv_out = conv_out * torch.sigmoid(conv_out)

        out[:, start:end] = conv_out

        if cache_indices is not None and state_len > 0:
            slot = int(cache_indices[b].item())
            conv_states[slot, :, :state_len] = padded[:, -state_len:]

    return out.to(original_x_dtype)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _seqlens_name(seqlens: list[int]) -> str:
    return "x".join(map(str, seqlens))


def _activation_name(activation: str | None) -> str:
    return activation if activation else "noact"


def _name_small(method, _idx, params):
    dtype, state_dtype, seqlens, width, activation, with_bias, with_initial_state = (
        params.args
    )
    bias_tag = "bias" if with_bias else "nobias"
    istate_tag = "istate" if with_initial_state else "no_istate"
    return (
        f"{method.__name__}_{_dtype_name(dtype)}_"
        f"state_{_dtype_name(state_dtype)}_"
        f"{_seqlens_name(seqlens)}_w{width}_"
        f"{_activation_name(activation)}_{bias_tag}_{istate_tag}"
    )


def _name_dtype_seqlens(method, _idx, params):
    dtype, seqlens = params.args
    return f"{method.__name__}_{_dtype_name(dtype)}_{_seqlens_name(seqlens)}"


def _normalize_activation(activation: str | bool | None) -> str:
    if activation is None:
        return ""
    if isinstance(activation, bool):
        return "silu" if activation else ""
    return activation


def _build_batch(
    seqlens: list[int],
    *,
    dim: int,
    width: int,
    num_cache_lines: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    device: torch.device,
    cache_indices: list[int] | None,
    has_initial_state: list[bool] | None,
) -> dict[str, torch.Tensor]:
    g = torch.Generator(device=device).manual_seed(sum(seqlens) * 1000 + dim)

    cu_seqlen = sum(seqlens)
    x = torch.randn(dim, cu_seqlen, dtype=dtype, device=device, generator=g) * 0.5
    weight = torch.randn(dim, width, dtype=dtype, device=device, generator=g) * 0.3
    bias = torch.randn(dim, dtype=dtype, device=device, generator=g) * 0.1
    conv_states = (
        torch.randn(num_cache_lines, dim, width - 1,
                    dtype=state_dtype, device=device, generator=g)
        * 0.4
    )

    qsl = [0]
    for sl in seqlens:
        qsl.append(qsl[-1] + sl)
    query_start_loc = torch.tensor(qsl, dtype=torch.int32, device=device)

    ci = (
        torch.tensor(cache_indices, dtype=torch.int32, device=device)
        if cache_indices is not None else None
    )
    his = (
        torch.tensor(has_initial_state, dtype=torch.bool, device=device)
        if has_initial_state is not None else None
    )

    return {
        "x": x,
        "weight": weight,
        "bias": bias,
        "conv_states": conv_states,
        "query_start_loc": query_start_loc,
        "cache_indices": ci,
        "has_initial_state": his,
    }


def _run_cpp_and_oracle(
    inputs: dict[str, torch.Tensor],
    *,
    activation: str | None,
    bias: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cs_for_cpp = inputs["conv_states"].clone()
    cs_for_oracle = inputs["conv_states"].clone()
    bias_arg = inputs["bias"] if bias else None

    out_cpp = torch.ops.zentorch.gdn_causal_conv1d_fn(
        inputs["x"], inputs["weight"], bias_arg,
        cs_for_cpp,
        inputs["query_start_loc"],
        inputs["cache_indices"],
        inputs["has_initial_state"],
        _normalize_activation(activation),
        int(PAD_SLOT_ID),
    )
    out_oracle = causal_conv1d_fn_oracle(
        inputs["x"], inputs["weight"], bias_arg,
        conv_states=cs_for_oracle,
        query_start_loc=inputs["query_start_loc"],
        cache_indices=inputs["cache_indices"],
        has_initial_state=inputs["has_initial_state"],
        activation=activation,
    )
    return out_cpp, cs_for_cpp, out_oracle, cs_for_oracle


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GDN_CausalConv1dFn(Zentorch_TestCase):

    @parameterized.expand(
        list(product(
            DTYPES, STATE_DTYPES, SEQLENS_VARLEN, WIDTHS,
            ACTIVATIONS, WITH_BIAS, WITH_INITIAL_STATE,
        )),
        name_func=_name_small,
    )
    def test_cpp_matches_oracle_small(
        self,
        dtype: torch.dtype,
        state_dtype: torch.dtype,
        seqlens: list[int],
        width: int,
        activation: str | None,
        with_bias: bool,
        with_initial_state: bool,
    ) -> None:
        cpu = torch.device("cpu")
        dim = 8
        batch = len(seqlens)
        inputs = _build_batch(
            seqlens,
            dim=dim,
            width=width,
            num_cache_lines=batch + 2,
            dtype=dtype,
            state_dtype=state_dtype,
            device=cpu,
            cache_indices=list(range(batch)),
            has_initial_state=[with_initial_state] * batch,
        )

        out_cpp, cs_cpp, out_oracle, cs_oracle = _run_cpp_and_oracle(
            inputs, activation=activation, bias=with_bias,
        )

        self.assertEqual(out_cpp.dtype, dtype)
        atol, rtol = default_tolerance(dtype, state_dtype)
        self.assertEqual(out_cpp, out_oracle, atol=atol, rtol=rtol)
        self.assertEqual(
            cs_cpp, cs_oracle,
            atol=atol, rtol=rtol,
            msg="conv_states (cache) mismatch cpp vs oracle",
        )

    @parameterized.expand(
        [(torch.bfloat16, sl) for sl in SEQLENS_QWEN],
        name_func=_name_dtype_seqlens,
    )
    def test_qwen35_4b_conv_dim(
        self,
        dtype: torch.dtype,
        seqlens: list[int],
    ) -> None:
        s = Qwen35_4B_GDN
        cpu = torch.device("cpu")
        batch = len(seqlens)
        inputs = _build_batch(
            seqlens,
            dim=s.conv_dim,
            width=s.conv_kernel_size,
            num_cache_lines=batch + 2,
            dtype=dtype,
            state_dtype=torch.float32,
            device=cpu,
            cache_indices=list(range(batch)),
            has_initial_state=[True, False, True][:batch] + [False] * (batch - 3 if batch > 3 else 0),
        )
        out_cpp, cs_cpp, out_oracle, cs_oracle = _run_cpp_and_oracle(
            inputs, activation="silu", bias=True,
        )
        atol, rtol = default_tolerance(dtype, torch.float32)
        self.assertEqual(out_cpp, out_oracle, atol=atol, rtol=rtol)
        self.assertEqual(cs_cpp, cs_oracle, atol=atol, rtol=rtol)

    def test_pad_slot_id_skips_sequence(self) -> None:
        cpu = torch.device("cpu")
        dim, width = 4, 4
        seqlens = [5, 3]
        inputs = _build_batch(
            seqlens,
            dim=dim,
            width=width,
            num_cache_lines=4,
            dtype=torch.float32,
            state_dtype=torch.float32,
            device=cpu,
            cache_indices=[0, PAD_SLOT_ID],
            has_initial_state=[True, True],
        )
        cs_before = inputs["conv_states"].clone()

        torch.ops.zentorch.gdn_causal_conv1d_fn(
            inputs["x"], inputs["weight"], inputs["bias"],
            inputs["conv_states"],
            inputs["query_start_loc"],
            inputs["cache_indices"],
            inputs["has_initial_state"],
            "silu",
            int(PAD_SLOT_ID),
        )

        self.assertFalse(torch.equal(inputs["conv_states"][0], cs_before[0]))
        for slot in (1, 2, 3):
            with self.subTest(slot=slot):
                self.assertTrue(
                    torch.equal(inputs["conv_states"][slot], cs_before[slot]),
                )


if __name__ == "__main__":
    run_tests()
