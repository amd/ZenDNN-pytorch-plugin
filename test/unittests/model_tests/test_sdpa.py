# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import math
import unittest
import torch
import sys
from pathlib import Path
from torch.nn.functional import scaled_dot_product_attention
from packaging.version import parse

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    SDPATestCase,
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
        self.assertEqual(
            native_output,
            zentorch_output,
            atol=1e-3,
            rtol=1e-2,
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
        self.assertEqual(
            ref_output,
            zen_output,
            atol=1e-3,
            rtol=1e-2,
            msg="Sliding-window output mismatch at long sequence length",
        )


if __name__ == "__main__":
    run_tests()
