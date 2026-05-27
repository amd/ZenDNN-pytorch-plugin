# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Tests for ``causal_conv1d_update`` (decode)."""

from __future__ import annotations

import sys
import unittest
from itertools import product
from pathlib import Path

import torch
from parameterized import parameterized

from test.unittests.op_tests.layers.gdn.helpers.shapes import Qwen35_4B_GDN
from test.unittests.op_tests.layers.gdn.helpers.varlen import NULL_BLOCK_ID, PAD_SLOT_ID

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from unittest_utils import (  # noqa: E402
    Zentorch_TestCase,
    default_tolerance,
    has_zentorch,
    run_tests,
)


DTYPES = [torch.float32, torch.bfloat16]
STATE_DTYPES = [torch.float32, torch.bfloat16]
BATCHES = [1, 4, 16]
SEQLENS = [None, 1, 4]
ACTIVATIONS = [None, "silu", "swish"]
WITH_BIAS = [True, False]
QWEN_BATCHES = [1, 16]


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
    width = weight.shape[1]
    windows = padded.unfold(dimension=-1, size=width, step=1)
    out = (windows * weight.unsqueeze(1)).sum(dim=-1)
    if bias is not None:
        out = out + bias.unsqueeze(-1)
    return out


def causal_conv1d_update_oracle(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | bool | None = None,
    *,
    conv_state_indices: torch.Tensor,
    null_block_id: int = NULL_BLOCK_ID,
    pad_slot_id: int = PAD_SLOT_ID,
) -> torch.Tensor:
    """Independent oracle for ``causal_conv1d_update``."""
    act = _resolve_activation(activation)

    is_2d = x.dim() == 2
    if is_2d:
        batch, dim = x.shape
        seqlen = 1
    else:
        batch, dim, seqlen = x.shape

    width = weight.shape[1]
    state_len = width - 1

    original_x_dtype = x.dtype
    compute_dtype = conv_state.dtype
    x_c = x.to(compute_dtype)
    if is_2d:
        x_c = x_c.unsqueeze(-1)
    w_c = weight.to(compute_dtype)
    b_c = bias.to(compute_dtype) if bias is not None else None

    out_3d = torch.empty_like(x_c)

    for b in range(batch):
        slot = int(conv_state_indices[b].item())
        if slot in (null_block_id, pad_slot_id):
            out_3d[b] = x_c[b]
            continue

        state = conv_state[slot, :, -state_len:].to(compute_dtype)
        x_b = x_c[b]

        padded = torch.cat([state, x_b], dim=-1)
        out_b = _depthwise_conv1d_manual(padded, w_c, b_c)

        if act == "silu":
            out_b = out_b * torch.sigmoid(out_b)

        out_3d[b] = out_b
        conv_state[slot, :, -state_len:] = padded[:, -state_len:].to(conv_state.dtype)

    out = out_3d.squeeze(-1) if is_2d else out_3d
    return out.to(original_x_dtype)


def _dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).split(".")[-1]


def _activation_name(activation: str | None) -> str:
    return activation if activation else "noact"


def _seqlen_tag(seqlen: int | None) -> str:
    if seqlen is None:
        return "x2d"
    return f"x3d_t{seqlen}"


def _name_small(method, _idx, params):
    dtype, state_dtype, batch, seqlen, activation, with_bias = params.args
    bias_tag = "bias" if with_bias else "nobias"
    return (
        f"{method.__name__}_{_dtype_name(dtype)}_"
        f"state_{_dtype_name(state_dtype)}_b{batch}_"
        f"{_seqlen_tag(seqlen)}_"
        f"{_activation_name(activation)}_{bias_tag}"
    )


def _name_dtype_batch(method, _idx, params):
    dtype, batch = params.args
    return f"{method.__name__}_{_dtype_name(dtype)}_b{batch}"


def _normalize_activation(activation: str | bool | None) -> str:
    if activation is None:
        return ""
    if isinstance(activation, bool):
        return "silu" if activation else ""
    return activation


def _build_inputs(
    *,
    batch: int,
    dim: int,
    width: int,
    seqlen: int | None,
    num_cache_lines: int,
    dtype: torch.dtype,
    state_dtype: torch.dtype,
    device: torch.device,
    cache_indices: list[int] | None = None,
    state_extra_len: int = 0,
) -> dict[str, torch.Tensor]:
    state_len = width - 1 + state_extra_len
    g = torch.Generator(device=device).manual_seed(batch * 1000 + dim + (seqlen or 0))

    if seqlen is None:
        x = torch.randn(batch, dim, dtype=dtype, device=device, generator=g) * 0.5
    else:
        x = torch.randn(batch, dim, seqlen, dtype=dtype, device=device, generator=g) * 0.5
    weight = torch.randn(dim, width, dtype=dtype, device=device, generator=g) * 0.3
    bias = torch.randn(dim, dtype=dtype, device=device, generator=g) * 0.1
    conv_state = (
        torch.randn(num_cache_lines, dim, state_len,
                    dtype=state_dtype, device=device, generator=g)
        * 0.4
    )
    if cache_indices is None:
        cache_indices = list(range(batch))
    ci = torch.tensor(cache_indices, dtype=torch.int32, device=device)

    return {
        "x": x,
        "weight": weight,
        "bias": bias,
        "conv_state": conv_state,
        "conv_state_indices": ci,
    }


def _run_cpp_and_oracle(
    inputs: dict[str, torch.Tensor],
    *,
    activation: str | None,
    bias: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cs_for_cpp = inputs["conv_state"].clone()
    cs_for_oracle = inputs["conv_state"].clone()
    bias_arg = inputs["bias"] if bias else None

    out_cpp = torch.ops.zentorch.gdn_causal_conv1d_update(
        inputs["x"], cs_for_cpp, inputs["weight"], bias_arg,
        _normalize_activation(activation),
        inputs["conv_state_indices"],
        int(NULL_BLOCK_ID),
        int(PAD_SLOT_ID),
    )
    out_oracle = causal_conv1d_update_oracle(
        inputs["x"],
        cs_for_oracle,
        inputs["weight"],
        bias_arg,
        activation,
        conv_state_indices=inputs["conv_state_indices"],
    )
    return out_cpp, cs_for_cpp, out_oracle, cs_for_oracle


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_GDN_CausalConv1dUpdate(Zentorch_TestCase):

    @parameterized.expand(
        list(product(DTYPES, STATE_DTYPES, BATCHES, SEQLENS, ACTIVATIONS, WITH_BIAS)),
        name_func=_name_small,
    )
    def test_cpp_matches_oracle_small(
        self,
        dtype: torch.dtype,
        state_dtype: torch.dtype,
        batch: int,
        seqlen: int | None,
        activation: str | None,
        with_bias: bool,
    ) -> None:
        cpu = torch.device("cpu")
        dim, width = 8, 4
        inputs = _build_inputs(
            batch=batch, dim=dim, width=width, seqlen=seqlen,
            num_cache_lines=batch + 2,
            dtype=dtype, state_dtype=state_dtype, device=cpu,
        )

        out_cpp, cs_cpp, out_oracle, cs_oracle = _run_cpp_and_oracle(
            inputs, activation=activation, bias=with_bias,
        )

        expected_shape = inputs["x"].shape
        self.assertEqual(out_cpp.shape, expected_shape)
        self.assertEqual(out_oracle.shape, expected_shape)
        self.assertEqual(out_cpp.dtype, dtype)
        self.assertEqual(out_oracle.dtype, dtype)

        atol, rtol = default_tolerance(dtype, state_dtype)
        self.assertEqual(out_cpp, out_oracle, atol=atol, rtol=rtol)
        self.assertEqual(
            cs_cpp, cs_oracle,
            atol=atol, rtol=rtol,
            msg="conv_state mismatch cpp vs oracle",
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
            batch=batch, dim=s.conv_dim, width=s.conv_kernel_size, seqlen=None,
            num_cache_lines=batch + 4,
            dtype=dtype, state_dtype=torch.float32, device=cpu,
        )
        out_cpp, cs_cpp, out_oracle, cs_oracle = _run_cpp_and_oracle(
            inputs, activation="silu", bias=True,
        )
        atol, rtol = default_tolerance(dtype, torch.float32)
        self.assertEqual(out_cpp, out_oracle, atol=atol, rtol=rtol)
        self.assertEqual(cs_cpp, cs_oracle, atol=atol, rtol=rtol)

    def test_pad_slot_and_null_block_skip(self) -> None:
        cpu = torch.device("cpu")
        dim, width, batch = 4, 4, 4
        inputs = _build_inputs(
            batch=batch, dim=dim, width=width, seqlen=None,
            num_cache_lines=8,
            dtype=torch.float32, state_dtype=torch.float32, device=cpu,
            cache_indices=[NULL_BLOCK_ID, 2, PAD_SLOT_ID, 3],
        )
        cs_before = inputs["conv_state"].clone()

        torch.ops.zentorch.gdn_causal_conv1d_update(
            inputs["x"], inputs["conv_state"], inputs["weight"], inputs["bias"],
            "silu",
            inputs["conv_state_indices"],
            int(NULL_BLOCK_ID),
            int(PAD_SLOT_ID),
        )

        self.assertFalse(torch.equal(inputs["conv_state"][2], cs_before[2]))
        self.assertFalse(torch.equal(inputs["conv_state"][3], cs_before[3]))
        for slot in (0, 1, 4, 5, 6, 7):
            with self.subTest(slot=slot):
                self.assertTrue(
                    torch.equal(inputs["conv_state"][slot], cs_before[slot]),
                )

    def test_state_update_matches_shift_left(self) -> None:
        cpu = torch.device("cpu")
        dim, width = 4, 4
        inputs = _build_inputs(
            batch=1, dim=dim, width=width, seqlen=None,
            num_cache_lines=2,
            dtype=torch.float32, state_dtype=torch.float32, device=cpu,
            cache_indices=[1],
        )
        state_before = inputs["conv_state"][1].clone()
        x_b = inputs["x"][0].clone()

        torch.ops.zentorch.gdn_causal_conv1d_update(
            inputs["x"], inputs["conv_state"], inputs["weight"], inputs["bias"],
            "",
            inputs["conv_state_indices"],
            int(NULL_BLOCK_ID),
            int(PAD_SLOT_ID),
        )

        expected_state = torch.stack([
            state_before[:, 1],
            state_before[:, 2],
            x_b,
        ], dim=-1)
        self.assertEqual(inputs["conv_state"][1], expected_state)


if __name__ == "__main__":
    run_tests()
