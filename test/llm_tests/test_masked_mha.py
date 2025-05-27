# ******************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Was sourced from
# https://github.com/intel/intel-extension-for-pytorch/blob/v2.3.0%2Bcpu/tests/cpu/test_rope.py
# IPEX commit ID: 339bd25
# ******************************************************************************

import unittest
from itertools import product
from typing import Tuple
import numpy as np
import torch
from parameterized import parameterized
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from llm_utils import ( # noqa: 402
    Zentorch_TestCase,
    run_tests,
    skip_test_pt_2_3,
    zentorch,
)  # noqa: 402

N = 2
batch_size_list = [2**i for i in range(N)]
head_num_kv_list = list({np.random.randint(1, 6) for _ in range(N)})
head_num_list = list(
    {np.lcm.reduce(head_num_kv_list) * np.random.randint(1, 3) for _ in range(N)}
)
beam_size_list = [1, 4]
head_size_list = list({np.random.randint(16, 256) for _ in range(N)})
max_seq_len_list = list({np.random.randint(32, 128) for _ in range(N)})


class Custom_Model_Masked_MHA(torch.nn.Module):
    def __init__(self, hidden_size=4096, n_head=16, n_head_kv=16, head_dim=256):
        super().__init__()
        self.num_heads = n_head
        self.num_kv = n_head_kv
        self.head_dim = head_dim
        self.query_key_value = nn.Linear(
            hidden_size, (n_head_kv * 2 + n_head) * head_dim
        )

    def _split_heads(
        self, fused_qkv: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split the last dimension into (num_heads, head_dim), results share
        same memory storage as `fused_qkv`

        Args:
            fused_qkv (`torch.tensor`, *required*): [batch_size, seq_length,
            (num_heads + kv_num * 2) * head_dim]

        Returns:
            query: [batch_size, seq_length, num_heads, head_dim]
            key: [batch_size, seq_length, kv_num, head_dim]
            value: [batch_size, seq_length, kv_num, head_dim]
        """
        bs = fused_qkv.shape[0]
        query_layer = fused_qkv[:, :, : self.num_heads * self.head_dim]
        query_layer = query_layer.view(bs, -1, self.num_heads, self.head_dim)
        key_layer = fused_qkv[
            :,
            :,
            self.num_heads
            * self.head_dim : (self.num_heads + self.num_kv)
            * self.head_dim,
        ]
        key_layer = key_layer.view(bs, -1, self.num_kv, self.head_dim)
        value_layer = fused_qkv[:, :, (self.num_heads + self.num_kv) * self.head_dim :]
        value_layer = value_layer.view(bs, -1, self.num_kv, self.head_dim)
        return query_layer, key_layer, value_layer

    def _repeat_kv(self, x: torch.Tensor, n_rep: int) -> torch.Tensor:
        "torch.repeat_interleave(x, dim=2, repeats=n_rep)"
        bs, slen, n_kv_heads, head_dim = x.shape
        if n_rep == 1:
            return x
        return (
            x[:, :, :, None, :]
            .expand(bs, slen, n_kv_heads, n_rep, head_dim)
            .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
        )

    def forward(
        self,
        input_t,
        key_cache,
        value_cache,
        max_position,
        attention_mask,
        beam_idx,
        indirect_access_kv_cache=False,
        offset=0,
        enable_linear=True,
    ):
        head_size = self.head_dim
        origin_type = input_t.dtype
        if enable_linear:
            query, key, value = self._split_heads(self.query_key_value(input_t))
        else:
            query, key, value = self._split_heads(input_t)
        if indirect_access_kv_cache:
            query = query.contiguous()
            key = key.contiguous()
            value = value.contiguous()
            return torch.ops.zentorch.zentorch_masked_multihead_self_attention(
                query,
                key,
                value,
                key_cache,
                value_cache,
                beam_idx,
                offset,
                head_size**0.5,
                max_position,
                None,
                attention_mask,
            )
        else:
            # Get the concatenated key and value
            if key_cache is not None:
                key = torch.cat([key_cache, key], dim=1)
                value = torch.cat([value_cache, value], dim=1)
            key_cache = key
            value_cache = value
            n_rep = self.num_heads // self.num_kv
            key = self._repeat_kv(key, n_rep)
            value = self._repeat_kv(value, n_rep)

            key = key.transpose(1, 2)
            query = query.transpose(1, 2)
            value = value.transpose(1, 2)
            if origin_type == torch.half:
                key = key.to(torch.float32)
                query = query.to(torch.float32)
                value = value.to(torch.float32)
            # matmul new_key and new_value to get the attention score
            attention_scores = torch.matmul(query, key.transpose(-1, -2))
            # scale the attention score
            attention_scores = attention_scores / (head_size**0.5)
            # import pdb; pdb.set_trace()
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            # softmax the attention score
            attention_probs = attention_scores.softmax(dim=-1)
            # matmul the attention score and value to get the context
            attention_output = torch.matmul(attention_probs, value)
            if origin_type == torch.half:
                attention_output = attention_output.to(origin_type)
            return attention_output, None, key_cache, value_cache, None


@unittest.skipIf(
    skip_test_pt_2_3, "Skipping test as OP support available from PyTorch 2.3"
)
class Test_Masked_MHA(Zentorch_TestCase):
    def setUp(self):
        super().setUp()
        self.first_seq_len = 32

    def assertEqual(self, x, y, prec=None, message="", allow_inf=False):
        if isinstance(prec, str) and message == "":
            message = prec
            prec = None
        if prec is None:
            prec = 1e-5

        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):

            def assertTensorsEqual(a, b):
                super(Zentorch_TestCase, self).assertEqual(a.size(), b.size(), message)
                if a.numel() > 0:
                    if a.device.type == "cpu" and (
                        a.dtype == torch.float16 or a.dtype == torch.bfloat16
                    ):
                        # CPU half and bfloat16 tensors don't have the methods
                        # we need below
                        a = a.to(torch.float32)
                    b = b.to(a)

                    if (a.dtype == torch.bool) != (b.dtype == torch.bool):
                        raise TypeError("Was expecting both tensors to be bool type.")
                    else:
                        if a.dtype == torch.bool and b.dtype == torch.bool:
                            # we want to respect precision but as bool doesn't
                            # support substraction,
                            # boolean tensor has to be converted to int
                            a = a.to(torch.int)
                            b = b.to(torch.int)

                        diff = a - b
                        if a.is_floating_point():
                            # check that NaNs are in the same locations
                            nan_mask = torch.isnan(a)
                            self.assertTrue(
                                torch.equal(nan_mask, torch.isnan(b)), message
                            )
                            diff[nan_mask] = 0
                            # inf check if allow_inf=True
                            if allow_inf:
                                inf_mask = torch.isinf(a)
                                inf_sign = inf_mask.sign()
                                self.assertTrue(
                                    torch.equal(inf_sign, torch.isinf(b).sign()),
                                    message,
                                )
                                diff[inf_mask] = 0
                        # TODO: implement abs on CharTensor (int8)
                        if diff.is_signed() and diff.dtype != torch.int8:
                            diff = diff.abs()
                        max_err = diff.max()
                        self.assertLessEqual(max_err, prec, message)

            super(Zentorch_TestCase, self).assertEqual(
                x.is_sparse, y.is_sparse, message
            )

            super(Zentorch_TestCase, self).assertEqual(
                x.is_quantized, y.is_quantized, message
            )

            assertTensorsEqual(x, y)
        else:
            super(Zentorch_TestCase, self).assertEqual(x, y, message)

    def _test_mha(
        self,
        beam_size,
        batch_size,
        head_size,
        head_num,
        head_num_kv,
        max_seq_len,
        first_seq_len,
    ):
        key_cache = None
        value_cache = None
        offset = 0
        mha = Custom_Model_Masked_MHA(
            hidden_size=head_num * head_size,
            n_head=head_num,
            n_head_kv=head_num_kv,
            head_dim=head_size,
        )
        torch._dynamo.reset()
        # Wrap the entire test in no_grad to ensure no gradients are computed
        with torch.no_grad():
            mha = torch.compile(mha, backend="zentorch")
            mha.eval()  # Ensure model is in eval mode

            # first token decode
            input_t = torch.randn(
                batch_size,
                first_seq_len,
                head_num * head_size,
                dtype=torch.float32,
                requires_grad=False,  # Explicitly disable gradients for input
            )
            key_cache_iakv = torch.randn(
                max_seq_len,
                beam_size * batch_size,
                head_num,
                head_size,
                dtype=torch.float32,
                requires_grad=False,
            )
            value_cache_iakv = torch.randn(
                max_seq_len,
                beam_size * batch_size,
                head_num,
                head_size,
                dtype=torch.float32,
                requires_grad=False,
            )
            beam_idx = torch.zeros(
                max_seq_len, beam_size * batch_size, dtype=torch.int64
            )
            # create attention mask and causal mask
            attention_mask = torch.zeros(
                batch_size,
                1,
                first_seq_len,
                first_seq_len,
                dtype=torch.float32,
                requires_grad=False,
            )
            casual_mask = torch.full(
                (first_seq_len, first_seq_len), -1e6, dtype=input_t.dtype
            )
            casual_mask = casual_mask.triu(1)
            casual_mask = casual_mask.unsqueeze(0).unsqueeze(0)
            attention_mask = (
                attention_mask + casual_mask
            )  # combine the attention mask and causal mask

            # UT for first token with fp32
            naive_output, _, key_cache, value_cache, _ = mha(
                input_t, None, None, max_seq_len, attention_mask, None, None
            )
            (
                indirect_access_kv_cache_output,
                _,
                key_cache_iakv,
                value_cache_iakv,
                beam_idx,
            ) = mha(
                input_t,
                key_cache_iakv,
                value_cache_iakv,
                max_seq_len,
                attention_mask,
                beam_idx,
                True,
                torch.tensor(offset),
            )

            key_cache = key_cache.repeat_interleave(beam_size, dim=0)
            value_cache = value_cache.repeat_interleave(beam_size, dim=0)
            for i in range(batch_size):
                self.assertEqual(
                    key_cache.transpose(0, 1)[:, i * beam_size, :, :],
                    key_cache_iakv[0:first_seq_len, i * beam_size, :, :],
                )
                self.assertEqual(
                    value_cache.transpose(0, 1)[:, i * beam_size, :, :],
                    value_cache_iakv[0:first_seq_len, i * beam_size, :, :],
                )
            if beam_size == 4:
                beam_idx_t = torch.zeros(beam_size * batch_size, dtype=torch.int64)
                for i in range(1, batch_size):
                    beam_idx_t[i * beam_size : i * beam_size + beam_size] = (
                        beam_idx_t[i * beam_size : i * beam_size + beam_size]
                        + i * beam_size
                    )
            elif beam_size == 1:
                beam_idx_t = torch.arange(batch_size)
            beam_idx[offset] = beam_idx_t
            # reorder cache for naive impelementation
            key_cache = torch.index_select(key_cache, 0, beam_idx_t)
            value_cache = torch.index_select(value_cache, 0, beam_idx_t)

            # UT for first token with bf16
            if zentorch._C.is_bf16_supported():
                input_t_bf16 = input_t.bfloat16()
                key_cache_iakv_bf16 = key_cache_iakv.bfloat16()
                value_cache_iakv_bf16 = value_cache_iakv.bfloat16()
                attention_mask_bf16 = attention_mask.bfloat16()
                with torch.autocast(
                    device_type="cpu",
                    enabled=True,
                    dtype=torch.bfloat16,
                ):
                    (
                        naive_output_bf16,
                        _,
                        key_cache_bf16,
                        value_cache_bf16,
                        _,
                    ) = mha(
                        input_t_bf16,
                        None,
                        None,
                        max_seq_len,
                        attention_mask_bf16,
                        None,
                        None,
                    )
                    (
                        indirect_access_kv_cache_output_bf16,
                        _,
                        key_cache_iakv_bf16,
                        value_cache_iakv_bf16,
                        beam_idx,
                    ) = mha(
                        input_t_bf16,
                        key_cache_iakv_bf16,
                        value_cache_iakv_bf16,
                        max_seq_len,
                        attention_mask_bf16,
                        beam_idx,
                        True,
                        torch.tensor(offset),
                    )
                    self.assertEqual(
                        naive_output_bf16,
                        indirect_access_kv_cache_output_bf16,
                        prec=2e-2,
                    )
                    key_cache_bf16 = key_cache_bf16.repeat_interleave(beam_size, dim=0)
                    value_cache_bf16 = value_cache_bf16.repeat_interleave(
                        beam_size, dim=0
                    )
                    for i in range(batch_size):
                        self.assertEqual(
                            key_cache_bf16.transpose(0, 1)[:, i * beam_size, :, :],
                            key_cache_iakv_bf16[0:first_seq_len, i * beam_size, :, :],
                        )
                        self.assertEqual(
                            value_cache_bf16.transpose(0, 1)[:, i * beam_size, :, :],
                            value_cache_iakv_bf16[0:first_seq_len, i * beam_size, :, :],
                        )
                    key_cache_bf16 = torch.index_select(key_cache_bf16, 0, beam_idx_t)
                    value_cache_bf16 = torch.index_select(
                        value_cache_bf16, 0, beam_idx_t
                    )

            offset = offset + first_seq_len
            # UT for next token with fp32
            input_t = torch.randn(
                beam_size * batch_size,
                1,
                head_num * head_size,
                dtype=torch.float32,
                requires_grad=False,
            )
            attention_mask = torch.zeros(
                beam_size * batch_size,
                1,
                1,
                offset + 1,
                dtype=torch.float32,
                requires_grad=False,
            )
            naive_output, _, key_cache, value_cache, _ = mha(
                input_t,
                key_cache,
                value_cache,
                max_seq_len,
                attention_mask,
                None,
                None,
            )
            (
                indirect_access_kv_cache_output,
                _,
                key_cache_iakv,
                value_cache_iakv,
                beam_idx,
            ) = mha(
                input_t,
                key_cache_iakv,
                value_cache_iakv,
                max_seq_len,
                attention_mask,
                beam_idx,
                True,
                torch.tensor(offset),
            )

            self.assertEqual(naive_output, indirect_access_kv_cache_output)
            self.assertEqual(
                key_cache.transpose(0, 1)[offset],
                key_cache_iakv[offset, :, :, :],
            )
            self.assertEqual(
                value_cache.transpose(0, 1)[offset],
                value_cache_iakv[offset, :, :, :],
            )
            # UT for next token with bf16
            if zentorch._C.is_bf16_supported():
                input_t_bf16 = input_t.bfloat16()
                attention_mask_bf16 = attention_mask.bfloat16()
                with torch.autocast(
                    device_type="cpu",
                    enabled=True,
                    dtype=torch.bfloat16,
                ):
                    (
                        naive_output_bf16,
                        _,
                        key_cache_bf16,
                        value_cache_bf16,
                        _,
                    ) = mha(
                        input_t_bf16,
                        key_cache_bf16,
                        value_cache_bf16,
                        max_seq_len,
                        attention_mask_bf16,
                        None,
                        None,
                    )
                    (
                        indirect_access_kv_cache_output_bf16,
                        _,
                        key_cache_iakv_bf16,
                        value_cache_iakv_bf16,
                        beam_idx,
                    ) = mha(
                        input_t_bf16,
                        key_cache_iakv_bf16,
                        value_cache_iakv_bf16,
                        max_seq_len,
                        attention_mask_bf16,
                        beam_idx,
                        True,
                        torch.tensor(offset),
                    )
                    self.assertEqual(
                        naive_output_bf16,
                        indirect_access_kv_cache_output_bf16,
                        prec=0.05,
                    )
                    self.assertEqual(
                        key_cache_bf16.transpose(0, 1)[offset],
                        key_cache_iakv_bf16[offset, :, :, :],
                    )
                    self.assertEqual(
                        value_cache_bf16.transpose(0, 1)[offset],
                        value_cache_iakv_bf16[offset, :, :, :],
                    )
                    if beam_size == 4:
                        beam_idx_t = torch.tensor([1, 3, 0, 0]).repeat(batch_size)
                        for i in range(1, batch_size):
                            beam_idx_t[i * beam_size : i * beam_size + beam_size] = (
                                beam_idx_t[i * beam_size : i * beam_size + beam_size]
                                + i * beam_size
                            )
                    elif beam_size == 1:
                        beam_idx_t = torch.arange(batch_size)
                    beam_idx[offset] = beam_idx_t
                    offset = offset + 1
                    # reorder cache for naive impelementation
                    key_cache = torch.index_select(key_cache, 0, beam_idx_t)
                    value_cache = torch.index_select(value_cache, 0, beam_idx_t)
                    key_cache_bf16 = torch.index_select(key_cache_bf16, 0, beam_idx_t)
                    value_cache_bf16 = torch.index_select(
                        value_cache_bf16, 0, beam_idx_t
                    )
            else:
                key_cache = torch.index_select(key_cache, 0, beam_idx_t)
                value_cache = torch.index_select(value_cache, 0, beam_idx_t)
                offset = offset + 1
            # UT for next token with fp32
            input_t = torch.randn(
                beam_size * batch_size,
                1,
                head_num * head_size,
                dtype=torch.float32,
                requires_grad=False,
            )
            attention_mask = torch.zeros(
                beam_size * batch_size,
                1,
                1,
                offset + 1,
                dtype=torch.float32,
                requires_grad=False,
            )
            naive_output, _, key_cache, value_cache, _ = mha(
                input_t,
                key_cache,
                value_cache,
                max_seq_len,
                attention_mask,
                None,
                None,
            )

            (
                indirect_access_kv_cache_output,
                _,
                key_cache_iakv,
                value_cache_iakv,
                beam_idx,
            ) = mha(
                input_t,
                key_cache_iakv,
                value_cache_iakv,
                max_seq_len,
                attention_mask,
                beam_idx,
                True,
                torch.tensor(offset),
            )
            self.assertEqual(naive_output, indirect_access_kv_cache_output)
            self.assertEqual(
                key_cache.transpose(0, 1)[offset],
                key_cache_iakv[offset, :, :, :],
            )
            self.assertEqual(
                value_cache.transpose(0, 1)[offset],
                value_cache_iakv[offset, :, :, :],
            )
            # UT for next token with bf16
            if zentorch._C.is_bf16_supported():
                input_t_bf16 = input_t.bfloat16()
                attention_mask_bf16 = attention_mask.bfloat16()
                with torch.autocast(
                    device_type="cpu",
                    enabled=True,
                    dtype=torch.bfloat16,
                ):
                    (
                        naive_output_bf16,
                        _,
                        key_cache_bf16,
                        value_cache_bf16,
                        _,
                    ) = mha(
                        input_t_bf16,
                        key_cache_bf16,
                        value_cache_bf16,
                        max_seq_len,
                        attention_mask_bf16,
                        None,
                        None,
                    )
                    (
                        indirect_access_kv_cache_output_bf16,
                        _,
                        key_cache_iakv_bf16,
                        value_cache_iakv_bf16,
                        beam_idx,
                    ) = mha(
                        input_t_bf16,
                        key_cache_iakv_bf16,
                        value_cache_iakv_bf16,
                        max_seq_len,
                        attention_mask_bf16,
                        beam_idx,
                        True,
                        torch.tensor(offset),
                    )
                    self.assertEqual(
                        naive_output_bf16,
                        indirect_access_kv_cache_output_bf16,
                        prec=0.05,
                    )
                    self.assertEqual(
                        key_cache_bf16.transpose(0, 1)[offset],
                        key_cache_iakv_bf16[offset, :, :, :],
                    )
                    self.assertEqual(
                        value_cache_bf16.transpose(0, 1)[offset],
                        value_cache_iakv_bf16[offset, :, :, :],
                    )

    @parameterized.expand(
        product(
            beam_size_list,
            batch_size_list,
            head_size_list,
            head_num_list,
            head_num_kv_list,
            max_seq_len_list,
        )
    )
    def test_llm_mha(
        self, beam_size, batch_size, head_size, head_num, head_num_kv, max_seq_len
    ):
        self._test_mha(
            beam_size,
            batch_size,
            head_size,
            head_num,
            head_num_kv,
            max_seq_len,
            self.first_seq_len,
        )


if __name__ == "__main__":
    run_tests()
