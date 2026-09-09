# ****************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ****************************************************************************

"""Override ``QwenGatedDeltaNetAttention.forward_cpu`` with the fp16 zentorch
path.

The override is dtype-aware: fp16 activations run the zentorch ``gdn_*``
kernels (vLLM's native CPU GDN path only supports bf16), while bf16 / fp32
fall straight through to vLLM's native ``forward_cpu``. Enabled by default;
set ``ZENTORCH_GDN=0`` to keep the native path for every dtype.

vLLM mamba / FLA submodules are imported lazily, after the deferred import
hook fires, so loading this module during plugin registration stays side-effect
free.

``vllm.utils.torch_utils`` is bound as a module object rather than via
``from ... import``: vLLM loads this plugin while that module is still
executing, so its names (``LayerNameType``) do not exist yet. Going through the
module defers every lookup -- including the ``LayerNameType`` annotation, which
torch's ``infer_schema`` eval()s against this module's globals at op
registration time.
"""

from __future__ import annotations

import os

import torch
from einops import rearrange

import vllm.utils.torch_utils as _vllm_torch_utils

from zentorch._logging import get_logger
from zentorch.vllm._import_hook import patch_now_or_on_import

logger = get_logger(__name__)

__all__ = ["_apply_gdn_patch"]

# vLLM 0.27 target: QwenGatedDeltaNetAttention (Qwen3-Next / Qwen3.5) is the
# GatedDeltaNet class that owns forward_cpu and binds it as _forward_method on
# CPU. The import hook patches the class before any instance is constructed, so
# __init__'s ``self._forward_method = self.forward_cpu`` binds the override.
_TARGET_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"

NULL_BLOCK_ID: int = 0
PAD_SLOT_ID: int = -1

_CORE_OP_REGISTERED = False


def _gdn_enabled() -> bool:
    """Enabled by default; set ``ZENTORCH_GDN=0`` to force vLLM's native path."""
    return os.environ.get("ZENTORCH_GDN", "1") != "0"


def _conv_weights(layer) -> torch.Tensor:
    """Return the plain ``(dim, width)`` conv weight for the zentorch kernels.

    On AMX builds vLLM VNNI-packs ``conv1d.weight`` in place at load time (only
    usable by its AMX kernel) and stashes the un-packed copy; prefer that when
    present. AMD EPYC has no AMX, so the ``.view`` path is the common case.
    """
    unpacked = getattr(layer.conv1d, "_cpu_unpacked_conv_weight", None)
    if unpacked is not None:
        return unpacked
    w = layer.conv1d.weight
    return w.view(w.size(0), w.size(2))


def _gdn_attention_core_cpu(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    core_attn_out: torch.Tensor,
    layer_name: _vllm_torch_utils.LayerNameType,
) -> None:
    """Core GDN attention (conv1d + gated delta rule) via zentorch fp16 ops.

    Registered as an opaque custom op so torch.compile treats it as a single
    external call and never specialises on the per-step integer fields of
    ``attn_metadata`` (num_decodes / num_prefills / token counts).
    """
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

    layer_name = _vllm_torch_utils._resolve_layer_name(layer_name)
    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    attn_metadata = forward_context.attn_metadata

    if attn_metadata is None:
        return

    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[layer.prefix]
    if not isinstance(attn_metadata, GDNAttentionMetadata):
        raise TypeError(
            "attn_metadata must be GDNAttentionMetadata; got "
            f"{type(attn_metadata).__name__}"
        )

    if attn_metadata.num_actual_tokens == 0:
        return

    # Explicit runtime checks (not ``assert``) so they survive ``python -O``.
    if (
        attn_metadata.spec_sequence_masks is not None
        or attn_metadata.num_accepted_tokens is not None
    ):
        raise NotImplementedError(
            "speculative decode is not supported in the zentorch fp16 CPU GDN "
            "attention path."
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
    # conv-state kernels expect the DS layout (num_slots, dim, state_len).
    conv_state = (
        layer_kv_cache[0]
        if is_conv_state_dim_first()
        else layer_kv_cache[0].transpose(-1, -2)
    )
    # zentorch gdn ops consume the raw (num_slots, num_v_heads, v_dim, k_dim)
    # ssm-state pool directly (see test_fused_recurrent_..._packed_decode); do
    # NOT apply vLLM's k_dim/v_dim view swap, which targets its own FLA kernels.
    ssm_state = layer_kv_cache[1]

    num_decodes = attn_metadata.num_decodes
    num_decode_tokens = attn_metadata.num_decode_tokens
    num_prefills = attn_metadata.num_prefills
    num_prefill_tokens = attn_metadata.num_prefill_tokens

    conv_weights = _conv_weights(layer)
    activation_str = layer.activation if layer.activation is not None else ""

    mixed_qkv = mixed_qkv.contiguous()
    a = a.contiguous()
    b = b.contiguous()

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

        prefill_mixed_qkv_t = prefill_mixed_qkv.transpose(0, 1)
        prefill_mixed_qkv = torch.ops.zentorch.gdn_causal_conv1d_fn(
            prefill_mixed_qkv_t,
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
            query,
            key,
            value,
            g,
            beta,
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

        # Recompute the FLA chunk metadata locally from the prefill-only
        # cu_seqlens derived above, using the same helpers vLLM uses. Keeps the
        # zentorch GDN path independent of how vLLM rebases the metadata's
        # chunk_indices / chunk_offsets across versions.
        from vllm.third_party.flash_linear_attention.ops.index import (
            prepare_chunk_indices,
            prepare_chunk_offsets,
        )
        from vllm.third_party.flash_linear_attention.ops.utils import (
            FLA_CHUNK_SIZE,
        )

        prefill_chunk_indices = prepare_chunk_indices(
            prefill_query_start_loc, FLA_CHUNK_SIZE
        )
        prefill_chunk_offsets = prepare_chunk_offsets(
            prefill_query_start_loc, FLA_CHUNK_SIZE
        )

        initial_state = ssm_state[prefill_state_indices].contiguous()
        initial_state[~prefill_has_initial_state, ...] = 0
        o, last_recurrent_state = (
            torch.ops.zentorch.gdn_chunk_gated_delta_rule_fwd(
                query,
                key,
                value,
                g,
                beta,
                float(layer.head_k_dim ** -0.5),
                initial_state,
                True,
                FLA_CHUNK_SIZE,
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
    layer_name: _vllm_torch_utils.LayerNameType,
) -> None:
    return None


def _register_core_op() -> None:
    """Register ``torch.ops.zentorch.gdn_attention_core_cpu`` (idempotent)."""
    global _CORE_OP_REGISTERED
    if _CORE_OP_REGISTERED or hasattr(
        torch.ops.zentorch, "gdn_attention_core_cpu"
    ):
        _CORE_OP_REGISTERED = True
        return

    from torch.library import Library

    # A FRAGMENT of the C++-defined "zentorch" library so the op lives in the
    # zentorch namespace alongside the gdn_* kernels it dispatches to. Kept at
    # module scope: the op's lifetime is tied to this Library object.
    global _zentorch_lib
    _zentorch_lib = Library("zentorch", "FRAGMENT")
    _vllm_torch_utils.direct_register_custom_op(
        op_name="gdn_attention_core_cpu",
        op_func=_gdn_attention_core_cpu,
        mutates_args=["core_attn_out"],
        fake_impl=_gdn_attention_core_cpu_fake,
        target_lib=_zentorch_lib,
    )
    _CORE_OP_REGISTERED = True


def forward_cpu_zen(
    self, hidden_states: torch.Tensor, output: torch.Tensor | None = None
) -> torch.Tensor | None:
    """Drop-in ``QwenGatedDeltaNetAttention.forward_cpu`` with an fp16 fast path.

    - **fp16 activations:** run the core GDN attention through the zentorch
      ``gdn_*`` kernels (vLLM's native CPU path only supports bf16).
    - **anything else (bf16 / fp32):** delegate to the original native
      ``forward_cpu`` unchanged.
    """
    if hidden_states.dtype != torch.float16:
        # Native path (bf16 / fp32) -- saved on the class by the patcher.
        return type(self)._zentorch_orig_forward_cpu(self, hidden_states)

    # Explicit runtime check (not ``assert``) so it survives ``python -O``.
    if hasattr(self, "in_proj_qkv"):
        raise NotImplementedError(
            "LoRA is not supported on the zentorch fp16 CPU GDN attention path."
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
        mixed_qkv,
        b,
        a,
        core_attn_out,
        _vllm_torch_utils._encode_layer_name(self.prefix),
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
    core_attn_out = core_attn_out.flatten(-2)  # ... h d -> ... (h d)

    out, _ = self.out_proj(core_attn_out)

    if output is None:
        # vLLM 0.27 contract: return the projected output directly.
        return out

    # Legacy contract: write into the caller-provided buffer.
    output[:num_tokens] = out
    return None


def _do_patch_gdn() -> bool:
    """Swap ``QwenGatedDeltaNetAttention.forward_cpu`` -> ``forward_cpu_zen``."""
    try:
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention,
        )
    except ImportError:
        logger.debug(
            "[zentorch] QwenGatedDeltaNetAttention not importable; skip GDN patch"
        )
        return False

    if getattr(QwenGatedDeltaNetAttention, "_zentorch_gdn_patched", False):
        return True

    _register_core_op()
    QwenGatedDeltaNetAttention._zentorch_orig_forward_cpu = (
        QwenGatedDeltaNetAttention.forward_cpu
    )
    QwenGatedDeltaNetAttention.forward_cpu = forward_cpu_zen
    QwenGatedDeltaNetAttention._zentorch_gdn_patched = True
    logger.info(
        "[zentorch] Patched QwenGatedDeltaNetAttention.forward_cpu "
        "-> forward_cpu_zen (fp16 GDN fast path; bf16/fp32 stay native)"
    )
    return True


def _apply_gdn_patch() -> bool:
    """Opt out with ZENTORCH_GDN=0 to keep vLLM's native CPU GDN attention."""
    if not _gdn_enabled():
        logger.debug(
            "[zentorch] GDN patch disabled (set ZENTORCH_GDN=0 to disable)"
        )
        return False
    return patch_now_or_on_import(_TARGET_MODULE, _do_patch_gdn)
