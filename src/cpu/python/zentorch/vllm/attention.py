# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from typing import Any, List
import torch

from zentorch._logging import get_logger

logger = get_logger(__name__)


class PagedAttention:
    """Implementation of zentorch PagedAttention - CPU decode path only"""

    def __repr__(self):
        return "PagedAttention(backend='zentorch', device='cpu')"

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 80, 96, 112, 128, 256]

    @staticmethod
    def validate_head_size(head_size: int) -> tuple[bool, list[int]]:
        """Validate if the head size is supported."""
        supported = PagedAttention.get_supported_head_sizes()
        return head_size in supported, supported

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int, block_size: int, num_kv_heads: int, head_size: int, *args
    ) -> tuple[int, ...]:
        return (2, num_blocks, block_size * num_kv_heads * head_size)

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor], src_to_dists: torch.Tensor | Any, *args
    ) -> None:
        key_caches = [kv[0] for kv in kv_caches]
        value_caches = [kv[1] for kv in kv_caches]
        # Assuming torch.ops._C_cache_ops.copy_blocks exists in the vLLM environment
        # If not, this fallback might need adjustment or removal depending on vLLM
        torch.ops._C_cache_ops.copy_blocks(key_caches, value_caches, src_to_dists)

    @staticmethod
    def split_kv_cache(
        kv_cache: torch.Tensor, num_kv_heads: int, head_size: int, *args
    ):
        num_blocks = kv_cache.shape[1]
        k = kv_cache[0].view(num_blocks, num_kv_heads, -1, head_size)
        v = kv_cache[1].view(num_blocks, num_kv_heads, -1, head_size)
        return k, v

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        kv_cache_dtype: str,
        k_scale: float,
        v_scale: float,
        *args,
    ) -> None:
        torch.ops.zentorch.zentorch_attention_reshape_and_cache(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping.flatten().int(),
        )

    @staticmethod
    def flash_attn_varlen_func(
        out: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
        softmax_scale: float,
        is_causal: bool,
        block_table: torch.Tensor,
        alibi_slopes: torch.Tensor | None,
        *,
        softcap: float = -1.0,
        window_size_left: int = -1,
        window_size_right: int = -1,
        kv_cache_dtype: str = "auto",
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        zentorch_op_name: str = "zentorch::zentorch_attention_flash_attn_varlen",
        **unused_kwargs,
    ) -> None:
        cu_seqlens_q = cu_seqlens_q.to(torch.int32, copy=False)
        cu_seqlens_k = cu_seqlens_k.to(torch.int32, copy=False)
        block_table = block_table.to(torch.int32, copy=False)
        softcap_value = -1.0
        if softcap is not None and softcap != 0.0:
            softcap_value = float(softcap)
        torch.ops.zentorch.zentorch_attention_flash_attn_varlen(
            out,
            query,
            key_cache,
            value_cache,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            float(softmax_scale),
            bool(is_causal),
            block_table,
            alibi_slopes,
            int(window_size_left),
            int(window_size_right),
            kv_cache_dtype,
            float(k_scale),
            float(v_scale),
            softcap_value,
            zentorch_op_name,
        )

    @staticmethod
    def forward_decode(
        output: torch.Tensor,
        query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        max_context_len: int,
        kv_cache_dtype: str,
        num_kv_heads: int,
        scale: float,
        alibi_slopes: torch.Tensor | None,
        k_scale: float,
        v_scale: float,
        *unused_args,
    ) -> None:
        # Assuming value_cache layout [1, num_blocks, block_size*num_heads*head_size]
        block_size = value_cache.shape[2]
        head_mapping = (
            torch.arange(
                0,
                num_kv_heads,
                device="cpu",
                dtype=torch.int32,
            )
            .view(num_kv_heads, 1)
            .repeat_interleave(query.size(1) // num_kv_heads)
            .flatten()
        )
        torch.ops.zentorch.zentorch_attention_single_query_cached_kv_attention(
            output,
            query.contiguous(),
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )
