# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import itertools
import math
import unittest
import torch
import sys
from pathlib import Path
from torch.nn.functional import scaled_dot_product_attention
from packaging.version import parse

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    DataTypes,
    SDPATestCase,
    Zentorch_TestCase,
    default_tolerance,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    update_supported_dtypes,
    seq_length_opt,
    batch_size_opt,
    mask_type_opt,
    num_heads_opt,
    head_dim_opt,
    gqa_head_config_opt,
    gqa_mask_type_opt,
    zentorch,
)

supported_dtypes = update_supported_dtypes(supported_dtypes, "zentorch_sdpa")


class Custom_Model_Sdpa(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sdpa = scaled_dot_product_attention

    def forward(self, query, key, value, attention_mask, scale):
        return self.sdpa(query, key, value, attn_mask=attention_mask, scale=scale)


class Test_Sdpa_Model(SDPATestCase):
    @SDPATestCase.hypothesis_params_sdpa_itr(
        dtype_list=supported_dtypes,
        seq_length_opt_list=seq_length_opt,
        batch_size_opt_list=batch_size_opt,
        mask_opt_list=mask_type_opt,
        num_heads_opt_list=num_heads_opt,
        head_dim_opt_list=head_dim_opt,
    )
    @torch.inference_mode()
    def test_sdpa_model(self, dtype, mask_type, head_dim):
        reset_dynamo()
        native_model = Custom_Model_Sdpa().eval()
        zentorch_model = Custom_Model_Sdpa().eval()
        zentorch_model = torch.compile(zentorch_model, backend="zentorch")
        with torch.inference_mode():
            sdpa_query = self.data.sdpa_query
            sdpa_key = self.data.sdpa_key
            sdpa_value = self.data.sdpa_value
            if mask_type == "none":
                sdpa_attention_mask = None
            else:
                mask_shape = self.data.mask_shape
                mask = torch.randint(0, 2, mask_shape, device="cpu").bool()
                if mask_type == "bfloat16" and dtype == "bfloat16":
                    sdpa_attention_mask = mask.to(torch.bfloat16)
                elif mask_type == "bool":
                    sdpa_attention_mask = mask
                else:
                    sdpa_attention_mask = mask.float()
            # Compute scale for attention: 1/sqrt(head_dim)
            scale = 1 / math.sqrt(head_dim)
            native_output = native_model(
                sdpa_query,
                sdpa_key,
                sdpa_value,
                sdpa_attention_mask,
                scale,
            )
            zentorch_output = zentorch_model(
                sdpa_query,
                sdpa_key,
                sdpa_value,
                sdpa_attention_mask,
                scale,
            )
            torch_version = torch.__version__
            # Parse the version
            parsed_version = parse(torch_version)
            if parsed_version.major == 2 and parsed_version.minor < 9:
                self.assertEqual(native_output, zentorch_output, atol=1e-2, rtol=1e-1)
            else:
                self.assertEqual(native_output, zentorch_output, atol=1e-3, rtol=1e-2)


def _make_gqa_tensors(
    batch,
    num_heads,
    kv_num_heads,
    seq_len_q,
    seq_len_kv,
    head_dim,
    dtype=torch.float32,
    tensor_seed=0,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(tensor_seed)
    query = torch.randn(
        batch, num_heads, seq_len_q, head_dim, generator=generator, dtype=dtype
    )
    key = torch.randn(
        batch, kv_num_heads, seq_len_kv, head_dim, generator=generator, dtype=dtype
    )
    value = torch.randn(
        batch, kv_num_heads, seq_len_kv, head_dim, generator=generator, dtype=dtype
    )
    return query, key, value


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
@unittest.skipIf(
    not zentorch._C.is_avx512_supported(),
    "zentorch_sdpa fp32 requires AVX512 on this hardware",
)
class Test_Sdpa_Gqa(SDPATestCase):
    @SDPATestCase.hypothesis_params_sdpa_itr(
        dtype_list=supported_dtypes,
        seq_length_opt_list=seq_length_opt,
        batch_size_opt_list=batch_size_opt,
        mask_opt_list=gqa_mask_type_opt,
        gqa_head_config_opt_list=gqa_head_config_opt,
        head_dim_opt_list=head_dim_opt,
    )
    @torch.inference_mode()
    def test_sdpa_gqa(self, dtype, mask_type, head_dim):
        query = self.data.sdpa_query
        key = self.data.sdpa_key
        value = self.data.sdpa_value
        num_heads = query.size(1)
        kv_num_heads = key.size(1)
        is_causal = mask_type == "causal"
        if mask_type == "float":
            attn_mask = torch.randn(self.data.mask_shape).to(query.dtype)
        else:
            attn_mask = None
        # Compute scale for attention: 1/sqrt(head_dim)
        scale = 1 / math.sqrt(head_dim)
        repeat = num_heads // kv_num_heads
        native_output = scaled_dot_product_attention(
            query,
            key.repeat_interleave(repeat, dim=1),
            value.repeat_interleave(repeat, dim=1),
            attn_mask=attn_mask,
            scale=scale,
            is_causal=is_causal,
            dropout_p=0.0,
        )
        zentorch_output, _ = torch.ops.zentorch.zentorch_sdpa(
            query,
            key,
            value,
            dropout_p=0.0,
            is_causal=is_causal,
            attn_mask=attn_mask,
            scale=scale,
        )
        atol, rtol = default_tolerance(DataTypes.get_torch_type(dtype))
        self.assertEqual(
            native_output,
            zentorch_output,
            atol=atol,
            rtol=rtol,
            msg=(
                f"GQA output mismatch for dtype={dtype}, mask={mask_type}, "
                f"num_heads={num_heads}, kv_num_heads={kv_num_heads}"
            ),
        )

    def test_sdpa_sliding_window_long_seq(self):
        """Gemma3-style sliding window; zentorch must not emit NaN on long seqs."""
        batch, num_heads, kv_num_heads, head_dim = 1, 3, 1, 256
        seq_len, sliding_window = 890, 257
        scale = 1.0 / math.sqrt(head_dim)
        query, key, value = _make_gqa_tensors(
            batch,
            num_heads,
            kv_num_heads,
            seq_len,
            seq_len,
            head_dim,
            dtype=torch.bfloat16,
            tensor_seed=3,
        )
        left = right = sliding_window - 1
        mask = torch.full((1, seq_len, seq_len), fill_value=1, dtype=torch.bfloat16)
        mask = torch.tril(mask, diagonal=right)
        mask = torch.triu(mask, diagonal=-left)
        mask = torch.log(mask).unsqueeze(1)

        repeat = num_heads // kv_num_heads
        ref_output = scaled_dot_product_attention(
            query,
            key.repeat_interleave(repeat, dim=1),
            value.repeat_interleave(repeat, dim=1),
            attn_mask=mask,
            scale=scale,
            dropout_p=0.0,
        )
        zen_output, _ = torch.ops.zentorch.zentorch_sdpa(
            query, key, value, dropout_p=0.0, attn_mask=mask, scale=scale
        )
        self.assertFalse(
            torch.isnan(zen_output).any().item(),
            "zentorch_sdpa produced NaN with sliding-window mask",
        )
        atol, rtol = default_tolerance(torch.bfloat16)
        self.assertEqual(
            ref_output,
            zen_output,
            atol=atol,
            rtol=rtol,
            msg="Sliding-window output mismatch at long sequence length",
        )


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
@unittest.skipIf(
    not zentorch._C.is_avx512_supported(),
    "zentorch_sdpa fp32 requires AVX512 on this hardware",
)
class Test_Sdpa_Out_Variant(Zentorch_TestCase):
    num_heads, kv_num_heads, seq_len, head_dim = 3, 1, 8, 16  # embeddinggemma-300m
    sliding_window = 4
    sentinel = 1024.0  # exactly representable in every dtype under test

    # Order in which the caller's buffer holds (batch, head, seq, head_dim) in
    # memory; the op is always handed it viewed back as [B, H, S, D].
    layouts = {
        "contiguous": (2, (0, 1, 2, 3)),
        "packed_batch": (2, (0, 2, 1, 3)),  # vLLM dense: packed [tokens, H, D]
        "per_sequence": (1, (0, 2, 1, 3)),  # vLLM ragged: one sequence slice
        "strided_head": (2, (0, 1, 3, 2)),  # head dim the kernels cannot store to
    }

    def _sliding_window_mask(self, dtype):
        """Symmetric bidirectional window, as vLLM builds it for Gemma3."""
        mask = torch.ones(1, 1, self.seq_len, self.seq_len, dtype=dtype)
        mask = torch.tril(mask, diagonal=self.sliding_window)
        return torch.log(torch.triu(mask, diagonal=-self.sliding_window))

    def test_out_writes_callers_buffer(self):
        dtypes = [torch.float32]
        if zentorch._C.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        cases = itertools.product(self.layouts.items(), dtypes, (False, True))

        for (layout, (batch, order)), dtype, masked in cases:
            with self.subTest(layout=layout, dtype=dtype, masked=masked):
                dims = (batch, self.num_heads, self.seq_len, self.head_dim)
                generator = torch.Generator(device="cpu").manual_seed(7)
                query, key, value = (
                    torch.randn(batch, heads, self.seq_len, self.head_dim,
                                generator=generator, dtype=dtype)
                    for heads in (self.num_heads,
                                  self.kv_num_heads, self.kv_num_heads)
                )
                mask = self._sliding_window_mask(dtype) if masked else None

                repeat = self.num_heads // self.kv_num_heads
                reference = scaled_dot_product_attention(
                    query,
                    key.repeat_interleave(repeat, dim=1),
                    value.repeat_interleave(repeat, dim=1),
                    attn_mask=mask,
                    dropout_p=0.0,
                )

                buffer = torch.full([dims[i] for i in order], self.sentinel,
                                    dtype=dtype)
                out = buffer.permute(*sorted(range(4), key=order.__getitem__))
                torch.ops.zentorch.zentorch_sdpa.out(
                    query, key, value, dropout_p=0.0, is_causal=False,
                    attn_mask=mask, scale=None, out=out,
                )

                self.assertFalse(
                    bool((buffer == self.sentinel).any().item()),
                    "zentorch_sdpa.out did not fill the caller's buffer",
                )
                self.assertEqual(out, reference, atol=1e-3, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
