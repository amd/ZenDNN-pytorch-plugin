# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

from typing import Any, List
import torch

from zentorch._logging import get_logger

logger = get_logger(__name__)


class PagedAttention:
    """Implementation of ZenTorch PagedAttention - CPU decode path only"""

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 80, 96, 112, 128, 256]

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


logger.info(
    "[vllm-zentorch] Using fallback Python PagedAttention implementation "
    "from zentorch_attention.py."
)
