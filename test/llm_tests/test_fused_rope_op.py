# ******************************************************************************
# Modifications Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
#
# Was sourced from
# https://github.com/intel/intel-extension-for-pytorch/blob/v2.7.0%2Bcpu/tests/cpu/test_rope.py
# IPEX commit ID: 30ecffa
# ******************************************************************************

import unittest
from itertools import product
import torch
from parameterized import parameterized
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from llm_utils import(  # noqa: 402
    Zentorch_TestCase,
    run_tests,
    skip_test_pt_2_3,
    set_seed,
    zentorch,
    freeze_opt,
    test_with_freeze_opt,
)

batch_sz_lst = [1, 2, 4, 8, 16, 32, 64]
seq_ln_lst = [32, 64, 128, 256, 512]


@unittest.skipIf(
    skip_test_pt_2_3, "Skipping test as OP support available from PyTorch 2.3"
)
class Test_Fused_Rope(Zentorch_TestCase):
    def setUp(self):
        super().setUp()
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

    @parameterized.expand(product(batch_sz_lst, seq_ln_lst, freeze_opt))
    def test_llm_rope(self, batch_sz, seq_ln, freeze_opt):
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

            # torch compile with zentorch backend
            torch._dynamo.reset()
            func_compile = torch.compile(func, backend="zentorch")
            query_compile_no_concat, _, _ = test_with_freeze_opt(
                func_compile,
                (
                    query,
                    embed_positions,
                    position_ids,
                    self.num_heads,
                    self.head_size,
                    offset,
                    rotary_dim,
                ),
                freeze_opt,
            )
            query_compile, key_compile, value_compile = test_with_freeze_opt(
                func_compile,
                (
                    linear_outs,
                    embed_positions,
                    position_ids,
                    self.num_heads,
                    self.head_size,
                    offset,
                    rotary_dim,
                ),
                freeze_opt,
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

    def test_deepseek_rope(self):
        def rotate_half(x):
            """Rotates half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
            cos = cos[position_ids].unsqueeze(unsqueeze_dim)
            sin = sin[position_ids].unsqueeze(unsqueeze_dim)
            b, h, s, d = q.shape
            q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
            b, h, s, d = k.shape
            k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
            q_embed = (q * cos) + (rotate_half(q) * sin)
            k_embed = (k * cos) + (rotate_half(k) * sin)
            return q_embed, k_embed

        def upcast_tensors(a: torch.Tensor, b: torch.Tensor):
            if a.dtype == torch.float and b.dtype != torch.float:
                return a, b.to(torch.float)
            elif a.dtype != torch.float and b.dtype == torch.float:
                return a.to(torch.float), b
            else:
                return a, b

        def upcast_and_assert(a: torch.Tensor, b: torch.Tensor, atol=1e-5):
            x, y = upcast_tensors(a, b)
            self.assertEqual(x, y, atol=atol, rtol=0)

        def deepseek_rope(
            q,
            kv,
            k_pe,
            sin,
            cos,
            position_ids,
            num_heads,
            q_head_dim,
            qk_nope_head_dim,
            qk_rope_head_dim,
            v_head_dim,
        ):
            bsz, q_len, _ = q.size()
            q = q.view(bsz, q_len, num_heads, q_head_dim).transpose(1, 2)
            q_nope, q_pe = torch.split(q, [qk_nope_head_dim, qk_rope_head_dim], dim=-1)
            k_pe = k_pe.view(bsz, q_len, 1, qk_rope_head_dim).transpose(1, 2)
            kv = kv.view(
                bsz, q_len, num_heads, qk_nope_head_dim + v_head_dim
            ).transpose(1, 2)
            k_nope, value_states = torch.split(
                kv, [qk_nope_head_dim, v_head_dim], dim=-1
            )
            q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)
            query_states = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)
            query_states[:, :, :, :qk_nope_head_dim] = q_nope
            query_states[:, :, :, qk_nope_head_dim:] = q_pe
            key_states = k_pe.new_empty(bsz, num_heads, q_len, q_head_dim)
            key_states[:, :, :, :qk_nope_head_dim] = k_nope
            key_states[:, :, :, qk_nope_head_dim:] = k_pe
            if q_head_dim != v_head_dim:
                value_states = torch.nn.functional.pad(
                    value_states, [0, q_head_dim - v_head_dim]
                )
            return query_states, key_states, value_states

        num_head = 128
        bs = 1
        seq_len = 10
        sincos = torch.rand(seq_len, num_head)
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)
        position_ids = torch.arange(seq_len).unsqueeze(0)
        q_head_dim = 192
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        v_head_dim = 128
        for dtype in [torch.float, torch.bfloat16]:
            prec = 1e-5 if dtype == torch.float32 else 5e-3
            enable_autocast = dtype == torch.bfloat16
            with torch.no_grad(), torch.amp.autocast("cpu", enabled=enable_autocast):
                q = torch.rand(bs, seq_len, num_head * q_head_dim, dtype=dtype)
                kv = torch.rand(
                    bs, seq_len, num_head, qk_nope_head_dim + v_head_dim, dtype=dtype
                )
                k_pe = torch.rand(bs, seq_len, 1, qk_rope_head_dim, dtype=dtype)
                q_clone = q.clone()
                kv_clone = kv.clone()
                k_pe_clone = k_pe.clone()
                q_clone2 = q.clone()
                k_pe_clone2 = k_pe.clone()
                q_ref, k_ref, v_ref = deepseek_rope(
                    q,
                    kv,
                    k_pe,
                    sin,
                    cos,
                    position_ids,
                    num_head,
                    q_head_dim,
                    qk_nope_head_dim,
                    qk_rope_head_dim,
                    v_head_dim,
                )
                q_z, k_z, v_z = torch.ops.zentorch.zentorch_rope_deepseek(
                    q_clone,
                    kv_clone,
                    k_pe_clone,
                    sincos,
                    position_ids,
                    num_head,
                    q_head_dim,
                    qk_nope_head_dim,
                    qk_rope_head_dim,
                )
                q_z2, k_pe_z2 = (
                    torch.ops.zentorch.zentorch_rope_deepseek_v2(
                        q_clone2,
                        k_pe_clone2,
                        sincos,
                        position_ids,
                        num_head,
                        q_head_dim,
                        qk_nope_head_dim,
                        qk_rope_head_dim,
                    )
                )
                upcast_and_assert(q_ref.transpose(1, 2), q_z, atol=prec)
                upcast_and_assert(k_ref.transpose(1, 2), k_z, atol=prec)
                upcast_and_assert(v_ref.transpose(1, 2), v_z, atol=prec)
                upcast_and_assert(q_ref.transpose(1, 2), q_z2, atol=prec)
                upcast_and_assert(k_ref.transpose(1, 2)[:, :, :1, qk_nope_head_dim:], k_pe_z2, atol=prec)


if __name__ == "__main__":
    run_tests()
