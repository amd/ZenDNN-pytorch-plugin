# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import math
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
    def test_sdpa_model(
        self, dtype, mask_type, head_dim
    ):
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


if __name__ == "__main__":
    run_tests()
