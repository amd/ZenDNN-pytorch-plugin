# ******************************************************************************
# Modifications Copyright (c) 2024 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Was sourced from
# https://github.com/intel/intel-extension-for-pytorch/blob/v2.3.0%2Bcpu/tests/cpu/test_rope.py
# IPEX commit ID: 339bd25
# ******************************************************************************

# add all the llm tests here
# TODO: disable some tests depending on PT version(s)

import numpy as np
import unittest
import torch
import torch.nn as nn
from itertools import product
from typing import Tuple
from parameterized import parameterized
import zentorch  # noqa
from torch.testing._internal.common_utils import TestCase, run_tests, SEED

skip_test_pt_2_3 = False
if torch.__version__[:3] < "2.3":
    skip_test_pt_2_3 = True

np.random.seed(SEED)
batch_sz_lst = [1, 2, 4, 8, 16, 32, 64]
seq_ln_lst = [32, 64, 128, 256, 512]

N = 2
batch_size_list = [2**i for i in range(N)]
head_num_kv_list = list({np.random.randint(1, 6) for _ in range(N)})
head_num_list = list(
    {np.lcm.reduce(head_num_kv_list) * np.random.randint(1, 3) for _ in range(N)}
)
beam_size_list = [1, 4]
head_size_list = list({np.random.randint(16, 256) for _ in range(N)})
max_seq_len_list = list({np.random.randint(32, 128) for _ in range(N)})


@unittest.skipIf(
    skip_test_pt_2_3, "Skipping test as OP support available from PyTorch 2.3"
)
class FusedROPETester(TestCase):
    def setUp(self):
        self.max_seq_len = 512
        # parameterizing over head_size and num_heads is not possible for now
        self.head_size = 256
        self.num_heads = 16
        self.hidden_size = self.head_size * self.num_heads

    def create_sinusoidal_positions(self, num_pos: int, dim: int) -> torch.Tensor:
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2) / dim))
        sinusoid_inp = torch.einsum(
            "i , j -> i j", torch.arange(num_pos, dtype=torch.float), inv_freq
        ).float()
        return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)

    @parameterized.expand(product(batch_sz_lst, seq_ln_lst))
    def test_rope(self, batch_sz, seq_ln):
        def _get_embed_positions(embed_positions, position_ids):
            if embed_positions.device != position_ids.device:
                embed_positions = embed_positions.to(position_ids.device)
                self.embed_positions = embed_positions
            return embed_positions.repeat(position_ids.shape[0], 1, 1)

        def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
            x1 = x[:, :, :, ::2]
            x2 = x[:, :, :, 1::2]
            x = torch.stack((-x2, x1), dim=-1)
            return x.flatten(-2)

        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(
            tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor, offset: int = 1
        ) -> torch.Tensor:
            if offset == 1:
                sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
                cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
                return (tensor * cos) + (rotate_every_two(tensor) * sin)
            else:
                sin = sin[:, :, None, :].repeat(1, 1, 1, 2)
                cos = cos[:, :, None, :].repeat(1, 1, 1, 2)
                return (tensor * cos) + (rotate_half(tensor) * sin)

        def func(
            input,
            embed_positions,
            position_ids,
            num_heads,
            head_size,
            offset,
            rotary_dim,
        ):
            return torch.ops.zentorch.zentorch_rope(
                input,
                embed_positions,
                position_ids,
                num_heads,
                head_size,
                offset,
                rotary_dim,
            )

        def hf_forward(
            query, key, position_ids, embed_positions, offset=None, rotary_dim=None
        ):
            embed_positions = _get_embed_positions(embed_positions, position_ids)
            repeated_position_ids = position_ids.unsqueeze(-1).repeat(
                1, 1, embed_positions.shape[-1]
            )
            sincos = torch.gather(embed_positions, 1, repeated_position_ids)
            sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

            if rotary_dim < self.head_size:
                k_rot = key[:, :, :, :rotary_dim]
                k_pass = key[:, :, :, rotary_dim:]

                q_rot = query[:, :, :, :rotary_dim]
                q_pass = query[:, :, :, rotary_dim:]

                k_rot = apply_rotary_pos_emb(k_rot, sin, cos, offset)
                q_rot = apply_rotary_pos_emb(q_rot, sin, cos, offset)

                key = torch.cat([k_rot, k_pass], dim=-1)
                query = torch.cat([q_rot, q_pass], dim=-1)
            else:
                key = apply_rotary_pos_emb(key, sin, cos, offset)
                query = apply_rotary_pos_emb(query, sin, cos, offset)
            return query, key

        def upcast_tensors(a: torch.Tensor, b: torch.Tensor):
            # only two dtypes are supported at the moment - bf16 and fp32,
            # so we can get away with this shortcut approach
            if a.dtype == torch.float and b.dtype != torch.float:
                return a, b.to(torch.float)
            elif a.dtype != torch.float and b.dtype == torch.float:
                return a.to(torch.float), b
            else:
                return a, b

        kv_heads = [self.num_heads, self.num_heads // 2]
        dtypes = [torch.float32, torch.bfloat16]
        position_ids_t = torch.arange(seq_ln).unsqueeze(0)
        position_ids_s = torch.Tensor([0]).to(torch.int64)
        model2rope_config = {
            "gptj": (64, 1, position_ids_t),
            "falcon": (self.head_size, 1, position_ids_s),
            "llama": (self.head_size, self.head_size // 2, position_ids_t),
            "gpt-neox": (24, 12, position_ids_t),
            "chatglm": (64, 1, position_ids_s),
            "codegen": (self.head_size, self.head_size // 2, position_ids_t),
        }
        for rope_config, kv_head, dtype in product(
            model2rope_config.values(), kv_heads, dtypes
        ):
            rotary_dim, offset, position_ids = rope_config
            # concat linear output
            linear_outs = torch.rand(
                batch_sz,
                seq_ln,
                self.hidden_size + kv_head * 2 * self.head_size,
            ).to(dtype)

            query = (
                linear_outs[:, :, : self.hidden_size]
                .contiguous()
                .view(batch_sz, seq_ln, self.num_heads, self.head_size)
            )
            key = (
                linear_outs[
                    :, :, self.hidden_size : self.hidden_size + kv_head * self.head_size
                ]
                .contiguous()
                .view(batch_sz, seq_ln, kv_head, self.head_size)
            )
            embed_positions = self.create_sinusoidal_positions(2048, rotary_dim)
            query_hf, key_hf = hf_forward(
                query, key, position_ids_t, embed_positions, offset, rotary_dim
            )
            # no concat q/k/v
            query_zentorch_no_concat, _, _ = torch.ops.zentorch.zentorch_rope(
                query,
                embed_positions,
                position_ids,
                self.num_heads,
                self.head_size,
                offset,
                rotary_dim,
            )
            key_zentorch_no_concat, _, _ = torch.ops.zentorch.zentorch_rope(
                key,
                embed_positions,
                position_ids,
                kv_head,
                self.head_size,
                offset,
                rotary_dim,
            )
            # concat q/k/v qkv_cocat -> ROPE -> (q, k, v)
            (
                query_zentorch,
                key_zentorch,
                value_zentorch,
            ) = torch.ops.zentorch.zentorch_rope(
                linear_outs,
                embed_positions,
                position_ids,
                self.num_heads,
                self.head_size,
                offset,
                rotary_dim,
            )

            # torch compile with zentorch backend.
            torch._dynamo.reset()
            func_compile = torch.compile(func, backend="zentorch")

            query_compile_no_concat, _, _ = func_compile(
                query,
                embed_positions,
                position_ids,
                self.num_heads,
                self.head_size,
                offset,
                rotary_dim,
            )
            query_compile, key_compile, value_compile = func_compile(
                linear_outs,
                embed_positions,
                position_ids,
                self.num_heads,
                self.head_size,
                offset,
                rotary_dim,
            )

            atol = 1e-5 if dtype == torch.float32 else 5e-3

            def upcast_and_assert(a: torch.Tensor, b: torch.Tensor, atol=1e-5):
                x, y = upcast_tensors(a, b)
                self.assertEqual(x, y, atol=atol, rtol=0)

            upcast_and_assert(query_compile_no_concat, query_hf, atol=atol)
            upcast_and_assert(query_compile, query_hf, atol=atol)
            upcast_and_assert(key_compile, key_hf, atol=atol)
            upcast_and_assert(query_hf, query_zentorch_no_concat, atol=atol)
            upcast_and_assert(key_hf, key_zentorch_no_concat, atol=atol)
            upcast_and_assert(query_hf, query_zentorch, atol=atol)
            upcast_and_assert(key_hf, key_zentorch, atol=atol)

            self.assertEqual(
                value_zentorch,
                linear_outs[:, :, self.hidden_size + kv_head * self.head_size :].view(
                    batch_sz, seq_ln, kv_head, self.head_size
                ),
            )
            self.assertEqual(
                value_compile,
                linear_outs[:, :, self.hidden_size + kv_head * self.head_size :].view(
                    batch_sz, seq_ln, kv_head, self.head_size
                ),
            )


class MaskedMHA(torch.nn.Module):
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
class MaskedMHATest(TestCase):
    def setUp(self):
        torch.manual_seed(SEED)
        self.first_seq_len = 32

    def assertEqual(self, x, y, prec=None, message="", allow_inf=False):
        if isinstance(prec, str) and message == "":
            message = prec
            prec = None
        if prec is None:
            prec = 1e-5

        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):

            def assertTensorsEqual(a, b):
                super(TestCase, self).assertEqual(a.size(), b.size(), message)
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

            super(TestCase, self).assertEqual(x.is_sparse, y.is_sparse, message)
            super(TestCase, self).assertEqual(x.is_quantized, y.is_quantized, message)
            assertTensorsEqual(x, y)
        else:
            super(TestCase, self).assertEqual(x, y, message)

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
        mha = MaskedMHA(
            hidden_size=head_num * head_size,
            n_head=head_num,
            n_head_kv=head_num_kv,
            head_dim=head_size,
        )
        torch._dynamo.reset()
        mha = torch.compile(mha, backend="zentorch")

        # first token decode
        input_t = torch.randn(
            batch_size,
            first_seq_len,
            head_num * head_size,
            dtype=torch.float32,
        )
        key_cache_iakv = torch.randn(
            max_seq_len,
            beam_size * batch_size,
            head_num,
            head_size,
            dtype=torch.float32,
        )
        value_cache_iakv = torch.randn(
            max_seq_len,
            beam_size * batch_size,
            head_num,
            head_size,
            dtype=torch.float32,
        )
        beam_idx = torch.zeros(max_seq_len, beam_size * batch_size, dtype=torch.int64)
        # create attention mask and causal mask
        attention_mask = torch.zeros(
            batch_size, 1, first_seq_len, first_seq_len, dtype=torch.float32
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
        with torch.inference_mode(), torch.no_grad():
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
            # self.assertEqual(naive_output,
            # indirect_access_kv_cache_output)
            key_cache = key_cache.repeat_interleave(beam_size, dim=0)
            value_cache = value_cache.repeat_interleave(beam_size, dim=0)
            for i in range(batch_size):
                self.assertEqual(
                    key_cache.transpose(0, 1)[:, i * beam_size, :, :],
                    key_cache_iakv[0 : first_seq_len, i * beam_size, :, :],
                )
                self.assertEqual(
                    value_cache.transpose(0, 1)[:, i * beam_size, :, :],
                    value_cache_iakv[0 : first_seq_len, i * beam_size, :, :],
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

        # # #UT for first token with bf16
        if zentorch._C.is_bf16_supported():
            input_t_bf16 = input_t.bfloat16()
            key_cache_iakv_bf16 = key_cache_iakv.bfloat16()
            value_cache_iakv_bf16 = value_cache_iakv.bfloat16()
            attention_mask_bf16 = attention_mask.bfloat16()
            with torch.inference_mode(), torch.no_grad(), torch.autocast(
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
                value_cache_bf16 = value_cache_bf16.repeat_interleave(beam_size, dim=0)
                for i in range(batch_size):
                    self.assertEqual(
                        key_cache_bf16.transpose(0, 1)[:, i * beam_size, :, :],  # no qa
                        key_cache_iakv_bf16[
                            0 : first_seq_len, i * beam_size, :, :
                        ],
                    )
                    self.assertEqual(
                        value_cache_bf16.transpose(0, 1)[:, i * beam_size, :, :],
                        value_cache_iakv_bf16[
                            0 : first_seq_len, i * beam_size, :, :
                        ],
                    )
                key_cache_bf16 = torch.index_select(key_cache_bf16, 0, beam_idx_t)
                value_cache_bf16 = torch.index_select(value_cache_bf16, 0, beam_idx_t)

        offset = offset + first_seq_len
        # UT for next token with fp32
        input_t = torch.randn(
            beam_size * batch_size,
            1,
            head_num * head_size,
            dtype=torch.float32,
        )
        attention_mask = torch.zeros(
            beam_size * batch_size, 1, 1, offset + 1, dtype=torch.float32
        )
        with torch.inference_mode(), torch.no_grad():
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
        # #UT for next token with bf16
        if zentorch._C.is_bf16_supported():
            input_t_bf16 = input_t.bfloat16()
            attention_mask_bf16 = attention_mask.bfloat16()
            with torch.inference_mode(), torch.no_grad(), torch.autocast(
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
                value_cache_bf16 = torch.index_select(value_cache_bf16, 0, beam_idx_t)
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
        )
        attention_mask = torch.zeros(
            beam_size * batch_size, 1, 1, offset + 1, dtype=torch.float32
        )
        with torch.inference_mode(), torch.no_grad():
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
        # #UT for next token with bf16
        if zentorch._C.is_bf16_supported():
            input_t_bf16 = input_t.bfloat16()
            attention_mask_bf16 = attention_mask.bfloat16()
            with torch.inference_mode(), torch.no_grad(), torch.autocast(
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
    def test_mha(
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
    print("Seed is", SEED)
    run_tests()
