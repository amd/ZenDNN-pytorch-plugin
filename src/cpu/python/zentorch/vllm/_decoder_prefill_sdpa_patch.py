# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************
"""Route Zen CPU prefill through zentorch_sdpa.out.

Mirrors https://github.com/vllm-project/vllm/pull/58298 for vLLM builds that
do not have that path yet. Equal sequence lengths use one dense ``zentorch_sdpa.out`` call. Unequal
fresh prefills and cached-prefix prefills use the same op with packed
``cu_seqlens`` offsets. A sliding window is forwarded as
``sliding_window_size``. Key blocks are unpacked with the layer ISA
(column-major ``[head_dim, block_size]`` for VEC and the other non-AMX
kernels); values stay token-major. AMX tiles stay on native attention.
Decode stays on the paged CPU kernel.

Set ``ZENTORCH_SDPA=0`` to opt out. A vLLM that already records
``_record_prefill_split`` is left alone.
"""

from __future__ import annotations

import copy

import torch

from zentorch._logging import get_logger
from zentorch.vllm._import_hook import patch_now_or_on_import
from zentorch.vllm._sdpa_utils import (
    zentorch_sdpa_enabled,
    zentorch_sdpa_supports_dtype,
)

logger = get_logger(__name__)

_TARGET_MODULE = "vllm.v1.attention.backends.cpu_attn"
_PATCH_MARKER = "_zentorch_prefill_sdpa_patched"


def should_use_zentorch_prefill_sdpa(
    attn_type: str,
    alibi_slopes: torch.Tensor | None,
    sliding_window: int | None,
    logits_soft_cap: float | None,
    sinks: torch.Tensor | None,
    is_fp8_kv_cache: bool,
    kv_sharing_target_layer_name: str | None,
    dtype: torch.dtype,
) -> bool:
    """True for plain decoder self-attention that zentorch_sdpa can run."""
    from vllm.v1.attention.backend import AttentionType

    if attn_type != AttentionType.DECODER or alibi_slopes is not None:
        return False
    if logits_soft_cap or sinks is not None:
        return False
    return zentorch_sdpa_supports_dtype(dtype)


def _reads_cached_prefix(attn_metadata) -> bool:
    """True when any prefill token must read a prefix already in the KV cache."""
    if getattr(attn_metadata, "num_extend_reqs", 0):
        return True
    prefill_loc = getattr(attn_metadata, "prefill_query_start_loc", None)
    if prefill_loc is None or prefill_loc.numel() < 2:
        return False
    offsets = [int(offset) for offset in prefill_loc.tolist()]
    query_lens = [
        end - start for start, end in zip(offsets[:-1], offsets[1:], strict=True)
    ]
    first_req = attn_metadata.num_native_reqs
    seq_lens = [
        int(length)
        for length in attn_metadata.seq_lens[
            first_req : first_req + len(query_lens)
        ].tolist()
    ]
    return query_lens != seq_lens


def _gather_cached(
    cache: torch.Tensor,
    block_row: torch.Tensor,
    token_start: int,
    n_tokens: int,
    out: torch.Tensor,
    *,
    column_major: bool,
) -> None:
    """Write ``n_tokens`` cached tokens, starting at ``token_start``, into ``out``.

    Values are stored token-major, ``[block_size, head_dim]`` per head. VEC,
    VEC16, NEON, RVV, VXE, and VSX store keys column-major,
    ``[head_dim, block_size]``. ``out`` is ``[n_tokens, heads, dim]``.
    """
    if n_tokens <= 0:
        return
    block_size = int(cache.size(2))
    heads, dim = out.size(1), out.size(2)
    head_stride = cache.stride(1)
    first_block, offset_in_block = divmod(token_start, block_size)
    remaining = n_tokens
    written = 0
    block_idx = first_block

    def _block_tokens(block_id: int) -> torch.Tensor:
        slab = cache[block_id]
        if column_major:
            laid = torch.as_strided(
                slab,
                size=(heads, dim, block_size),
                stride=(head_stride, block_size, 1),
            )
            return laid.permute(2, 0, 1)
        return slab.transpose(0, 1)

    while remaining > 0 and block_idx < block_row.shape[0]:
        take = min(block_size - offset_in_block, remaining)
        tokens = _block_tokens(int(block_row[block_idx]))
        out[written : written + take].copy_(
            tokens[offset_in_block : offset_in_block + take]
        )
        written += take
        remaining -= take
        offset_in_block = 0
        block_idx += 1


def _pack_kv(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    start_loc: list[int],
    seq_lens: list[int],
    block_table: torch.Tensor,
    *,
    key_column_major: bool,
    sliding_window_size: int = 0,
    is_causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[int]] | None:
    """Pack K/V and return the token count actually stored for each request.

    An extend's prefix is gathered from the cache into one buffer and its
    current chunk is copied in after that prefix. A causal sliding window
    only needs the last ``window - 1`` prefix tokens. A fresh prefill is the
    current chunk only. Returns ``None`` when a block row cannot hold
    ``prefix_len``; a shorter offset list cannot stand in for the missing
    tokens. The returned lengths are what ``cu_seqlens_k`` must use.
    """
    spans: list[tuple[int, int, int, int]] = []
    total = 0
    for i, seq_len in enumerate(seq_lens):
        q0, q1 = start_loc[i], start_loc[i + 1]
        query_len = q1 - q0
        prefix_len = seq_len - query_len
        block_row = block_table[i]
        token_start = 0
        n_prefix = 0
        if prefix_len > 0:
            if block_row.numel() == 0:
                return None
            for cache in (key_cache, value_cache):
                block_size = int(cache.size(2))
                if int(block_row.shape[0]) * block_size < prefix_len:
                    return None
            if is_causal and sliding_window_size > 0:
                n_prefix = min(prefix_len, max(sliding_window_size - 1, 0))
            else:
                n_prefix = prefix_len
            token_start = prefix_len - n_prefix
        spans.append((q0, q1, token_start, n_prefix))
        total += n_prefix + query_len

    if len(spans) == 1 and spans[0][3] == 0:
        q0, q1 = spans[0][0], spans[0][1]
        return key[q0:q1], value[q0:q1], [q1 - q0]

    packed_key = key.new_empty((total, key.size(1), key.size(2)))
    packed_value = value.new_empty((total, value.size(1), value.size(2)))
    k_lens: list[int] = []
    offset = 0
    for i, (q0, q1, token_start, n_prefix) in enumerate(spans):
        if n_prefix:
            _gather_cached(
                key_cache,
                block_table[i],
                token_start,
                n_prefix,
                packed_key[offset : offset + n_prefix],
                column_major=key_column_major,
            )
            _gather_cached(
                value_cache,
                block_table[i],
                token_start,
                n_prefix,
                packed_value[offset : offset + n_prefix],
                column_major=False,
            )
        end = offset + n_prefix + (q1 - q0)
        packed_key[offset + n_prefix : end].copy_(key[q0:q1])
        packed_value[offset + n_prefix : end].copy_(value[q0:q1])
        k_lens.append(end - offset)
        offset = end
    return packed_key, packed_value, k_lens


def _uniform_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    num_reqs: int,
    seq_len: int,
    scale: float,
    sliding_window_size: int,
    is_causal: bool,
) -> None:
    """Equal-length fresh prefill: packed tokens to one dense SDPA."""

    def _packed_to_bhsd(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unflatten(0, (num_reqs, seq_len)).permute(0, 2, 1, 3)

    torch.ops.zentorch.zentorch_sdpa.out(
        _packed_to_bhsd(query),
        _packed_to_bhsd(key),
        _packed_to_bhsd(value),
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
        sliding_window_size=sliding_window_size,
        out=_packed_to_bhsd(output),
    )


def _varlen_out(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens_q: list[int],
    output: torch.Tensor,
    scale: float,
    cu_seqlens_k: list[int] | None = None,
    sliding_window_size: int = 0,
    is_causal: bool = True,
) -> None:
    torch.ops.zentorch.zentorch_sdpa.out(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        is_causal=is_causal,
        scale=scale,
        sliding_window_size=sliding_window_size,
        out=output,
    )


def zentorch_prefill_sdpa(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    attn_metadata,
    scale: float,
    sliding_window: int | None = None,
    is_causal: bool = True,
    key_column_major: bool = True,
) -> bool:
    """Fill the prefill suffix of ``output``. Decode rows are left untouched.

    Returns ``False`` when a cached prefix is not fully present in the block
    table. The caller must run native attention for that request.
    """
    # None and -1 are "no window". A positive width is the ZenDNN width.
    window = 0 if sliding_window is None or sliding_window < 0 else sliding_window
    query_start_loc = getattr(attn_metadata, "prefill_query_start_loc", None)
    if query_start_loc is None or query_start_loc.numel() < 2:
        return True

    start_loc = [int(offset) for offset in query_start_loc.tolist()]
    query_lens = [
        end - start for start, end in zip(start_loc[:-1], start_loc[1:], strict=True)
    ]
    if not query_lens:
        return True

    num_extend_reqs = attn_metadata.num_extend_reqs
    first_req = attn_metadata.num_native_reqs
    first_token = attn_metadata.num_native_tokens
    end_token = first_token + start_loc[-1]
    num_prefill_reqs = len(query_lens)
    seq_lens = [int(length) for length in attn_metadata.seq_lens[first_req:].tolist()][
        :num_prefill_reqs
    ]

    prefill_query = query[first_token:end_token]
    prefill_key = key[first_token:end_token]
    prefill_value = value[first_token:end_token]
    prefill_output = output[first_token:end_token]
    if num_extend_reqs == 0 and query_lens == seq_lens and len(set(seq_lens)) == 1:
        _uniform_sdpa(
            prefill_query,
            prefill_key,
            prefill_value,
            prefill_output,
            num_prefill_reqs,
            seq_lens[0],
            scale,
            window,
            is_causal,
        )
        return True

    if num_extend_reqs == 0 and query_lens == seq_lens:
        _varlen_out(
            prefill_query,
            prefill_key,
            prefill_value,
            start_loc,
            prefill_output,
            scale,
            sliding_window_size=window,
            is_causal=is_causal,
        )
        return True

    packed = _pack_kv(
        prefill_key,
        prefill_value,
        key_cache,
        value_cache,
        start_loc,
        seq_lens,
        attn_metadata.block_table[first_req : first_req + num_prefill_reqs],
        key_column_major=key_column_major,
        sliding_window_size=window,
        is_causal=is_causal,
    )
    if packed is None:
        return False
    packed_key, packed_value, k_lens = packed
    cu_k = [0]
    for length in k_lens:
        cu_k.append(cu_k[-1] + length)
    _varlen_out(
        prefill_query,
        packed_key,
        packed_value,
        start_loc,
        prefill_output,
        scale,
        cu_seqlens_k=cu_k,
        sliding_window_size=window,
        is_causal=is_causal,
    )
    return True


def _model_dtype() -> torch.dtype:
    try:
        from vllm.config import get_current_vllm_config_or_none
    except ImportError:
        from vllm.config import (
            get_current_vllm_config as get_current_vllm_config_or_none,
        )

    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is None:
        return torch.bfloat16
    return vllm_config.model_config.dtype


def _group_uses_zentorch_prefill_sdpa(builder) -> bool:
    from vllm.config import get_layers_from_vllm_config
    from vllm.model_executor.layers.attention import Attention

    layers = get_layers_from_vllm_config(
        builder.vllm_config, Attention, builder.layer_names
    )
    return any(
        getattr(layer.impl, "use_zentorch_prefill_sdpa", False)
        for layer in layers.values()
    )


def _record_prefill_split(builder, attn_metadata, common_attn_metadata) -> None:
    from vllm import _custom_ops as ops
    from vllm import envs
    from vllm.v1.attention.backends.utils import split_decodes_prefills_and_extends

    if (
        attn_metadata.dynamic_causal is not None
        or common_attn_metadata.seq_lens_cpu_upper_bound is None
    ):
        return

    decode_threshold = builder.reorder_batch_threshold or 1
    (
        _,
        num_extends,
        num_fresh_prefills,
        _,
        extend_tokens,
        fresh_prefill_tokens,
    ) = split_decodes_prefills_and_extends(
        common_attn_metadata, decode_threshold=decode_threshold
    )
    num_prefill_reqs = int(num_extends) + int(num_fresh_prefills)
    if not num_prefill_reqs:
        return
    num_prefill_tokens = int(extend_tokens) + int(fresh_prefill_tokens)
    num_reqs = common_attn_metadata.num_reqs
    num_native_reqs = num_reqs - num_prefill_reqs
    num_native_tokens = common_attn_metadata.num_actual_tokens - num_prefill_tokens
    query_start_loc = attn_metadata.query_start_loc

    attn_metadata.num_extend_reqs = int(num_extends)
    attn_metadata.num_native_reqs = num_native_reqs
    attn_metadata.num_native_tokens = num_native_tokens
    attn_metadata.prefill_query_start_loc = (
        query_start_loc[num_native_reqs : num_reqs + 1] - num_native_tokens
    )
    if num_native_reqs:
        attn_metadata.native_scheduler_metadata = ops.cpu_attn_get_scheduler_metadata(
            num_reqs=num_native_reqs,
            num_heads=builder.num_heads,
            num_kv_heads=builder.num_kv_heads,
            head_dim=builder.head_dim,
            seq_lens=attn_metadata.seq_lens[:num_native_reqs],
            dtype=builder.dtype,
            query_start_loc=query_start_loc[: num_native_reqs + 1],
            causal=attn_metadata.causal,
            sliding_window_size=builder.window_size,
            isa=builder.isa,
            enable_kv_split=envs.VLLM_CPU_ATTN_SPLIT_KV,
            dynamic_causal=attn_metadata.dynamic_causal,
            kv_cache_dtype=builder.kv_cache_dtype,
        )


def _do_patch_prefill_sdpa() -> bool:
    from vllm.v1.attention.backends.cpu_attn import (
        CPUAttentionBackendImpl,
        CPUAttentionMetadataBuilder,
    )

    if hasattr(CPUAttentionMetadataBuilder, "_record_prefill_split"):
        logger.debug("[zentorch] vLLM already routes prefill through zentorch_sdpa")
        return False
    if getattr(CPUAttentionBackendImpl, _PATCH_MARKER, False):
        return True

    orig_impl_init = CPUAttentionBackendImpl.__init__
    orig_forward = CPUAttentionBackendImpl.forward
    orig_builder_init = CPUAttentionMetadataBuilder.__init__
    orig_build = CPUAttentionMetadataBuilder.build

    def _impl_init(self, *args, **kwargs):
        orig_impl_init(self, *args, **kwargs)
        self.use_zentorch_prefill_sdpa = should_use_zentorch_prefill_sdpa(
            attn_type=self.attn_type,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            sinks=self.sinks,
            is_fp8_kv_cache=self.is_fp8_kv_cache,
            kv_sharing_target_layer_name=self.kv_sharing_target_layer_name,
            dtype=_model_dtype(),
        )

    def _forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        *args,
        **kwargs,
    ):
        prefill_loc = getattr(attn_metadata, "prefill_query_start_loc", None)
        # Encoder-only attention (EmbeddingGemma) does not bind a paged
        # decoder cache, so kv_cache stays a rank-1 placeholder.
        if (
            attn_metadata is not None
            and self.attn_type == "decoder"
            and getattr(self, "use_zentorch_prefill_sdpa", False)
            and prefill_loc is not None
            and not (
                _reads_cached_prefix(attn_metadata)
                and getattr(layer, "isa", "vec") in ("amx", "amx_fp8")
            )
        ):
            num_blocks, num_kv_heads, block_size, _ = kv_cache.size()
            key_cache, value_cache = kv_cache.view(
                (num_blocks, num_kv_heads, block_size * 2, -1)
            ).chunk(2, dim=2)
            if zentorch_prefill_sdpa(
                query,
                key,
                value,
                output,
                key_cache,
                value_cache,
                attn_metadata,
                self.scale,
                self.sliding_window,
                attn_metadata.causal,
                key_column_major=getattr(layer, "isa", "vec") not in ("amx", "amx_fp8"),
            ):
                if attn_metadata.num_native_reqs == 0:
                    return output
                attn_metadata = copy.copy(attn_metadata)
                attn_metadata.num_actual_tokens = attn_metadata.num_native_tokens
                attn_metadata.query_start_loc = attn_metadata.query_start_loc[
                    : attn_metadata.num_native_reqs + 1
                ]
                attn_metadata.seq_lens = attn_metadata.seq_lens[
                    : attn_metadata.num_native_reqs
                ]
                attn_metadata.block_table = attn_metadata.block_table[
                    : attn_metadata.num_native_reqs
                ]
                attn_metadata.scheduler_metadata = (
                    attn_metadata.native_scheduler_metadata
                )
                attn_metadata.prefill_query_start_loc = None
        return orig_forward(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            *args,
            **kwargs,
        )

    def _builder_init(self, *args, **kwargs):
        orig_builder_init(self, *args, **kwargs)
        self.use_zentorch_prefill_sdpa = _group_uses_zentorch_prefill_sdpa(self)
        if self.use_zentorch_prefill_sdpa and hasattr(
            self, "_init_reorder_batch_threshold"
        ):
            self._init_reorder_batch_threshold(1, supports_spec_as_decode=True)

    def _build(self, common_prefix_len, common_attn_metadata, fast_build=False):
        attn_metadata = orig_build(
            self, common_prefix_len, common_attn_metadata, fast_build
        )
        if getattr(self, "use_zentorch_prefill_sdpa", False):
            _record_prefill_split(self, attn_metadata, common_attn_metadata)
        return attn_metadata

    CPUAttentionBackendImpl.__init__ = _impl_init
    CPUAttentionBackendImpl.forward = _forward
    CPUAttentionMetadataBuilder.__init__ = _builder_init
    CPUAttentionMetadataBuilder.build = _build
    setattr(CPUAttentionBackendImpl, _PATCH_MARKER, True)
    logger.info(
        "[zentorch] Patched CPUAttentionBackendImpl " "-> zentorch_sdpa.out (prefill)"
    )
    return True


def _apply_prefill_sdpa_patch() -> bool:
    """Opt out with ``ZENTORCH_SDPA=0``."""
    if not zentorch_sdpa_enabled():
        logger.debug("[zentorch] prefill SDPA patch disabled or op unavailable")
        return False
    return patch_now_or_on_import(_TARGET_MODULE, _do_patch_prefill_sdpa)
