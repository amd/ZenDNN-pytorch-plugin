# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import math
import os
import random
from typing import Tuple

import torch

from llm_utils import Zentorch_TestCase, run_tests, set_seed
from zentorch.vllm.attention import PagedAttention


def _fill_sliding_mask_(
    mask: torch.Tensor | None,
    q_size: int,
    k_size: int,
    window_size_left: int,
    window_size_right: int,
    dtype: torch.dtype,
    device: torch.device,
):
    if mask is None:
        mask = torch.zeros(q_size, k_size, dtype=dtype, device=device)
    for row in range(q_size):
        idx = k_size - q_size + row
        if window_size_left > 0 and idx - window_size_left > 0:
            mask[row][: idx - window_size_left] = float("-inf")
        if window_size_right > 0 and idx + window_size_right + 1 < k_size:
            mask[row][idx + window_size_right + 1 :] = float("-inf")
    return mask


def _mha_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    *,
    is_causal: bool,
    window_size: Tuple[int, int],
    softcap: float,
) -> torch.Tensor:
    dtype = torch.float32
    device = q.device
    q_float = q.to(dtype)
    k_float = k.to(dtype)
    v_float = v.to(dtype)

    if is_causal:
        cur_mask = torch.full(
            (q_float.size(-2), q_float.size(-2)),
            float("-inf"),
            dtype=dtype,
            device=device,
        ).triu(1)
        past_mask = torch.zeros(
            q_float.size(-2),
            k_float.size(-2) - q_float.size(-2),
            dtype=dtype,
            device=device,
        )
        mask = torch.cat([past_mask, cur_mask], dim=-1)
    else:
        mask = None

    if window_size != (-1, -1):
        mask = _fill_sliding_mask_(
            mask,
            q_float.size(-2),
            k_float.size(-2),
            window_size[0],
            window_size[1],
            dtype=dtype,
            device=device,
        )

    kv_groups = q_float.size(1) // k_float.size(1)
    k_float = k_float.repeat_interleave(kv_groups, dim=1)
    v_float = v_float.repeat_interleave(kv_groups, dim=1)

    attn = torch.matmul(q_float, k_float.transpose(-2, -1)) * scale
    if softcap != -1:
        attn = torch.tanh(attn / softcap) * softcap
    if mask is not None:
        attn = attn + mask

    attn = torch.softmax(attn, dim=-1)
    output = torch.matmul(attn, v_float)
    return output.to(q.dtype)


class TestFlashAttnVarLen(Zentorch_TestCase):
    def _run_flash_attn_varlen_case(
        self,
        *,
        dtype: torch.dtype,
        is_causal: bool,
        window_size: Tuple[int, int],
        softcap: float,
        num_heads: int = 8,
        num_queries_per_kv: int = 2,
        head_size: int = 64,
        page_size: int = 16,
        max_query_len: int = 48,
        max_ctx_len: int = 32,
        kv_cache_dtype: str = "auto",
        k_scale: float = 1.0,
        v_scale: float = 1.0,
    ) -> None:
        set_seed(0)
        random.seed(0)

        batch_size = 3
        num_kv_heads = num_heads // num_queries_per_kv

        query_lens = [random.randint(1, max_query_len) for _ in range(batch_size)]
        ctx_lens = [random.randint(0, max_ctx_len) for _ in range(batch_size)]
        seq_lens = [q_len + c_len for q_len, c_len in zip(query_lens, ctx_lens, strict=True)]

        cu_seqlens_q = torch.tensor([0] + query_lens, dtype=torch.int32).cumsum(0)
        cu_seqlens_k = torch.tensor([0] + seq_lens, dtype=torch.int32).cumsum(0)

        max_seq_len_q = int(max(query_lens))
        max_seq_len_k = int(max(seq_lens))
        max_blocks_per_request = math.ceil(max_seq_len_k / page_size)
        num_pages = batch_size * max_blocks_per_request + 8

        block_table = torch.empty(batch_size, max_blocks_per_request, dtype=torch.int32)
        block_id = 0
        for i in range(batch_size):
            for j in range(max_blocks_per_request):
                block_table[i, j] = block_id
                block_id += 1

        total_queries = int(sum(query_lens))
        total_tokens = int(sum(seq_lens))

        query_base = torch.randn(
            total_queries, num_heads, head_size, dtype=torch.float32
        )
        key_base = torch.randn(
            total_tokens, num_kv_heads, head_size, dtype=torch.float32
        )
        value_base = torch.randn(
            total_tokens, num_kv_heads, head_size, dtype=torch.float32
        )

        query = query_base.to(dtype)
        key_cache = torch.zeros(
            num_pages, num_kv_heads, page_size, head_size, dtype=dtype
        )
        value_cache = torch.zeros_like(key_cache)

        for batch_idx in range(batch_size):
            seq_len = int(seq_lens[batch_idx])
            seq_blocks = math.ceil(seq_len / page_size)
            k_start = int(cu_seqlens_k[batch_idx].item())
            k_end = int(cu_seqlens_k[batch_idx + 1].item())
            key_slice = key_base[k_start:k_end]
            value_slice = value_base[k_start:k_end]

            for block_idx in range(seq_blocks):
                global_block_id = int(block_table[batch_idx, block_idx].item())
                token_start = block_idx * page_size
                token_end = min(token_start + page_size, seq_len)
                token_range = slice(token_start, token_end)
                key_cache[global_block_id, :, : token_end - token_start, :] = (
                    key_slice[token_range].transpose(0, 1).to(dtype)
                )
                value_cache[global_block_id, :, : token_end - token_start, :] = (
                    value_slice[token_range].transpose(0, 1).to(dtype)
                )

        scale = float(1.0 / math.sqrt(head_size))
        output = torch.empty_like(query)

        PagedAttention.flash_attn_varlen_func(
            output,
            query,
            key_cache,
            value_cache,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seq_len_q,
            max_seq_len_k,
            scale,
            is_causal,
            block_table,
            None,
            softcap=softcap,
            window_size_left=window_size[0],
            window_size_right=window_size[1],
            kv_cache_dtype=kv_cache_dtype,
            k_scale=k_scale,
            v_scale=v_scale,
            zentorch_op_name="zentorch::zentorch_attention_flash_attn_varlen",
        )

        output_ref = torch.empty(
            total_queries, num_heads, head_size, dtype=torch.float32
        )
        for batch_idx in range(batch_size):
            q_start = int(cu_seqlens_q[batch_idx].item())
            q_end = int(cu_seqlens_q[batch_idx + 1].item())
            k_start = int(cu_seqlens_k[batch_idx].item())
            k_end = int(cu_seqlens_k[batch_idx + 1].item())

            query_i = query_base[q_start:q_end]
            key_i = key_base[k_start:k_end]
            value_i = value_base[k_start:k_end]

            output_i = _mha_ref(
                query_i.unsqueeze(0).transpose(1, 2),
                key_i.unsqueeze(0).transpose(1, 2),
                value_i.unsqueeze(0).transpose(1, 2),
                scale,
                is_causal=is_causal,
                window_size=window_size,
                softcap=softcap,
            )
            output_i = output_i.squeeze(0).transpose(0, 1)
            output_ref[q_start:q_end] = output_i

        ref_cast = output_ref.to(torch.float32)
        out_cast = output.to(torch.float32)

        if dtype == torch.float32:
            atol, rtol = 5e-4, 1e-3
        else:
            atol, rtol = 5e-2, 5e-2

        self.assertTrue(torch.allclose(out_cast, ref_cast, atol=atol, rtol=rtol))

    def test_flash_attn_varlen(self) -> None:
        configs = [
            {
                "dtype": torch.float32,
                "is_causal": True,
                "window_size": (-1, -1),
                "softcap": -1.0,
            },
            {
                "dtype": torch.float32,
                "is_causal": False,
                "window_size": (2, 2),
                "softcap": 25.0,
            },
            {
                "dtype": torch.float32,
                "is_causal": True,
                "window_size": (4, -1),
                "softcap": -1.0,
                "page_size": 32,
                "max_query_len": 32,
                "max_ctx_len": 48,
            },
            {
                "dtype": torch.float32,
                "is_causal": False,
                "window_size": (-1, -1),
                "softcap": 5.0,
                "num_queries_per_kv": 1,
                "head_size": 48,
                "max_query_len": 40,
                "max_ctx_len": 64,
            },
            {
                "dtype": torch.bfloat16,
                "is_causal": True,
                "window_size": (-1, -1),
                "softcap": -1.0,
                "num_queries_per_kv": 4,
                "page_size": 32,
            },
            {
                "dtype": torch.bfloat16,
                "is_causal": False,
                "window_size": (0, 6),
                "softcap": 15.0,
                "head_size": 128,
                "max_query_len": 64,
                "max_ctx_len": 96,
            },
        ]

        # Test both GEMM paths: matmul_direct (1) and tensor-based (0)
        env_values = [
            ("zendnn_matmul_direct", "1"),
            ("zendnn_matmul_tensor", "0"),
        ]

        for path_name, env_val in env_values:
            os.environ["USE_ZENDNN_SDPA_MATMUL_DIRECT"] = env_val
            for cfg in configs:
                with self.subTest(gemm_path=path_name, cfg=cfg):
                    self._run_flash_attn_varlen_case(**cfg)


if __name__ == "__main__":
    run_tests()
