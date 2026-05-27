# ****************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""CPU forward for ``GatedDeltaNetAttention`` backed by zentorch C++ ops."""

from __future__ import annotations

import torch
from einops import rearrange
from torch._inductor.lowering import make_fallback
from torch.library import Library

from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn_linear_attn import (
    GDNAttentionMetadata,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first

NULL_BLOCK_ID: int = 0
PAD_SLOT_ID: int = -1
FLA_CHUNK_SIZE: int = 64

__all__ = ["forward_cpu_zen"]


# The heavy lifting is wrapped as a registered torch op so torch.compile /
# Dynamo treats it as a single opaque call, preventing specialisation of
# attn_metadata integer fields at warmup.
_lib = Library("zentorch", "FRAGMENT")
_lib.define(
    "gdn_attention_core_cpu("
    "Tensor mixed_qkv, Tensor b, Tensor a, "
    "Tensor(a!) core_attn_out, str layer_name"
    ") -> ()"
)


def _gdn_attention_core_cpu(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    attn_metadata_raw = forward_context.attn_metadata

    if attn_metadata_raw is None:
        return

    if isinstance(attn_metadata_raw, dict):
        attn_metadata = attn_metadata_raw[layer.prefix]
    else:
        attn_metadata = attn_metadata_raw
    if not isinstance(attn_metadata, GDNAttentionMetadata):
        raise TypeError(
            "attn_metadata must be GDNAttentionMetadata; got "
            f"{type(attn_metadata).__name__}"
        )

    if attn_metadata.num_actual_tokens == 0:
        return

    # Explicit runtime check (not ``assert``) so the guard survives ``python -O``.
    if (
        attn_metadata.spec_sequence_masks is not None
        or attn_metadata.num_accepted_tokens is not None
    ):
        raise NotImplementedError(
            "speculative decode not supported in CPU GDN attention (zentorch)."
        )

    state_indices_tensor = attn_metadata.non_spec_state_indices_tensor
    query_start_loc = attn_metadata.non_spec_query_start_loc
    has_initial_state = attn_metadata.has_initial_state
    if state_indices_tensor is None:
        raise RuntimeError(
            "attn_metadata.non_spec_state_indices_tensor must not be None"
        )
    if query_start_loc is None:
        raise RuntimeError(
            "attn_metadata.non_spec_query_start_loc must not be None"
        )

    layer_kv_cache = layer.kv_cache
    conv_state = (
        layer_kv_cache[0]
        if is_conv_state_dim_first()
        else layer_kv_cache[0].transpose(-1, -2)
    )
    ssm_state = layer_kv_cache[1]

    num_decodes = attn_metadata.num_decodes
    num_decode_tokens = attn_metadata.num_decode_tokens
    num_prefills = attn_metadata.num_prefills
    num_prefill_tokens = attn_metadata.num_prefill_tokens

    conv_weights = layer.conv1d.weight.view(
        layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)
    )
    activation_str = layer.activation if layer.activation is not None else ""

    if num_decodes > 0:
        decode_mixed_qkv = mixed_qkv[:num_decode_tokens]
        decode_b = b[:num_decode_tokens]
        decode_a = a[:num_decode_tokens]
        decode_state_indices = state_indices_tensor[:num_decodes]

        decode_mixed_qkv = torch.ops.zentorch.gdn_causal_conv1d_update(
            decode_mixed_qkv,
            conv_state,
            conv_weights,
            layer.conv1d.bias,
            activation_str,
            decode_state_indices,
            NULL_BLOCK_ID,
            PAD_SLOT_ID,
        )

        out_buf = core_attn_out[:num_decode_tokens].unsqueeze(1)
        torch.ops.zentorch.gdn_fused_recurrent_gated_delta_rule_packed_decode(
            decode_mixed_qkv,
            decode_a,
            decode_b,
            layer.A_log,
            layer.dt_bias,
            layer.head_k_dim ** -0.5,
            ssm_state,
            out_buf,
            decode_state_indices,
            True,
        )

    if num_prefills > 0:
        if has_initial_state is None:
            raise RuntimeError(
                "attn_metadata.has_initial_state must not be None when "
                "num_prefills > 0"
            )

        prefill_token_start = num_decode_tokens
        prefill_token_end = prefill_token_start + num_prefill_tokens
        prefill_mixed_qkv = mixed_qkv[prefill_token_start:prefill_token_end]
        prefill_b = b[prefill_token_start:prefill_token_end]
        prefill_a = a[prefill_token_start:prefill_token_end]
        prefill_state_indices = state_indices_tensor[
            num_decodes : num_decodes + num_prefills
        ]
        prefill_query_start_loc = (
            query_start_loc[num_decodes : num_decodes + num_prefills + 1]
            - num_decode_tokens
        )
        prefill_has_initial_state = has_initial_state[
            num_decodes : num_decodes + num_prefills
        ]

        prefill_mixed_qkv_T = prefill_mixed_qkv.transpose(0, 1)
        prefill_mixed_qkv = torch.ops.zentorch.gdn_causal_conv1d_fn(
            prefill_mixed_qkv_T,
            conv_weights,
            layer.conv1d.bias,
            conv_state,
            prefill_query_start_loc,
            prefill_state_indices,
            prefill_has_initial_state,
            activation_str,
            PAD_SLOT_ID,
        ).transpose(0, 1)

        (
            query, key, value, g, beta,
        ) = torch.ops.zentorch.gdn_fused_post_conv_prep(
            prefill_mixed_qkv,
            prefill_a,
            prefill_b,
            layer.A_log,
            layer.dt_bias,
            layer.num_k_heads // layer.tp_size,
            layer.head_k_dim,
            layer.head_v_dim,
            True,
            False,
        )
        query = query.unsqueeze(0)
        key = key.unsqueeze(0)
        value = value.unsqueeze(0)
        g = g.unsqueeze(0)
        beta = beta.unsqueeze(0)

        # Rebase chunk_indices / chunk_offsets onto the prefill-only slice;
        # vLLM computes them against the full non-spec batch.
        if num_decodes > 0:
            prefill_chunk_offsets = (
                attn_metadata.chunk_offsets[num_decodes:] - num_decodes
            )
            prefill_chunk_indices = (
                attn_metadata.chunk_indices[num_decodes:].clone()
            )
            prefill_chunk_indices[:, 0] -= num_decodes
        else:
            prefill_chunk_indices = attn_metadata.chunk_indices
            prefill_chunk_offsets = attn_metadata.chunk_offsets

        initial_state = ssm_state[prefill_state_indices].contiguous()
        initial_state[~prefill_has_initial_state, ...] = 0
        o, last_recurrent_state = (
            torch.ops.zentorch.gdn_chunk_gated_delta_rule_fwd(
                query, key, value, g, beta,
                float(layer.head_k_dim ** -0.5),
                initial_state, True, FLA_CHUNK_SIZE,
                prefill_query_start_loc,
                prefill_chunk_indices,
                prefill_chunk_offsets,
            )
        )
        o = o.to(query.dtype)

        ssm_state[prefill_state_indices] = last_recurrent_state.to(
            ssm_state.dtype
        )
        core_attn_out[prefill_token_start:prefill_token_end] = o.squeeze(0)


def _gdn_attention_core_cpu_fake(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: str,
) -> None:
    return None


_lib.impl("gdn_attention_core_cpu", _gdn_attention_core_cpu, "CPU")
_lib.impl("gdn_attention_core_cpu", _gdn_attention_core_cpu_fake, "Meta")

# Inductor fallback so any torch.compile region that crosses
# forward_cpu_zen can emit this as an external call instead of failing
# with "missing lowering" (mirrors the make_fallback pass in
# _meta_registrations.py for the C++ gdn_* ops). Cannot live in
# _meta_registrations.py because that file is loaded at zentorch
# package-init time, before the deferred-patch hook runs forward.py
# and the op gets registered.
make_fallback(torch.ops.zentorch.gdn_attention_core_cpu)


def forward_cpu_zen(self, hidden_states: torch.Tensor, output: torch.Tensor) -> None:
    """Drop-in replacement for ``GatedDeltaNetAttention.forward_cpu``."""
    # Explicit runtime check (not ``assert``) so the guard survives ``python -O``.
    if hasattr(self, "in_proj_qkv"):
        raise NotImplementedError(
            "LoRA is not supported on CPU GDN attention (zentorch)."
        )

    mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
    ba, _ = self.in_proj_ba(hidden_states)

    if self.gqa_interleaved_layout:
        query, key, value, z, b, a = self.fix_query_key_value_ordering(
            mixed_qkvz, ba
        )
        query, key, value = (
            rearrange(x, "l p d -> l (p d)") for x in (query, key, value)
        )
        mixed_qkv = torch.cat((query, key, value), dim=-1)
    else:
        qkv_size = (self.key_dim * 2 + self.value_dim) // self.tp_size
        z_size = self.value_dim // self.tp_size
        mixed_qkv, z = mixed_qkvz.split([qkv_size, z_size], dim=-1)
        z = z.reshape(z.size(0), -1, self.head_v_dim)
        b, a = ba.chunk(2, dim=-1)

    num_tokens = hidden_states.size(0)
    core_attn_out = torch.zeros(
        (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    torch.ops.zentorch.gdn_attention_core_cpu(
        mixed_qkv, b, a, core_attn_out, self.prefix,
    )

    z_shape_og = z.shape
    core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    core_attn_out = torch.ops.zentorch.gdn_rms_norm_gated(
        core_attn_out,
        self.norm.weight,
        z,
        self.norm.eps,
        getattr(self.norm, "activation", "swish"),
    )
    core_attn_out = core_attn_out.reshape(z_shape_og)

    core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
    output[:num_tokens], _ = self.out_proj(core_attn_out)
