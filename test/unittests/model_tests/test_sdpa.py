# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import math
import torch
from parameterized import parameterized
import sys
from pathlib import Path
from torch.nn.functional import scaled_dot_product_attention
from itertools import product

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    seq_length_opt,
    batch_size_opt,
    mask_type_opt,
    num_heads_opt,
    head_dim_opt,
)


class Custom_Model_Sdpa(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sdpa = scaled_dot_product_attention

    def forward(self, query, key, value, attention_mask, scale):
        return self.sdpa(query, key, value, attn_mask=attention_mask, scale=scale)


class Test_Sdpa_Model(Zentorch_TestCase):
    @parameterized.expand(
        product(
            supported_dtypes,
            seq_length_opt,
            batch_size_opt,
            mask_type_opt,
            num_heads_opt,
            head_dim_opt,
        )
    )
    @torch.inference_mode()
    def test_sdpa_model(
        self, dtype, seq_length, batch_size, mask_type, num_heads, head_dim
    ):
        self.data.create_unittest_data(dtype)
        torch_type = self.data.get_torch_type(dtype)
        reset_dynamo()
        native_model = Custom_Model_Sdpa().eval()
        zentorch_model = Custom_Model_Sdpa().eval()
        zentorch_model = torch.compile(zentorch_model, backend="zentorch")
        with torch.inference_mode():
            sdpa_query = torch.randn(
                batch_size,
                num_heads,
                seq_length,
                head_dim,
                device="cpu",
                requires_grad=False,
            ).type(torch_type)
            sdpa_key = torch.randn(
                batch_size,
                num_heads,
                seq_length,
                head_dim,
                device="cpu",
                requires_grad=False,
            ).type(torch_type)
            sdpa_value = torch.randn(
                batch_size,
                num_heads,
                seq_length,
                head_dim,
                device="cpu",
                requires_grad=False,
            ).type(torch_type)
            if mask_type == "none":
                sdpa_attention_mask = None
            else:
                mask_shape = (batch_size, num_heads, seq_length, seq_length)
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
            self.assertEqual(native_output, zentorch_output, atol=1e-3, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
