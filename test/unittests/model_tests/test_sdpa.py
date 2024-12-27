# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
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
)

skip_test_pt_2_4 = False

if torch.__version__[:3] < "2.4":
    skip_test_pt_2_4 = True


class Custom_Model_Sdpa(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.sdpa = scaled_dot_product_attention

    def forward(self, query, key, value, attention_mask):
        # Scale= 1/sqrt(hidden_size_per_head) and here we considered head size as 64,
        # Hence scale 0.125 is used
        # TODO Update test case with
        # parameterizing the num_heads and hidden_size_per_head
        return self.sdpa(query, key, value, attn_mask=attention_mask, scale=0.125)


@unittest.skipIf(
    skip_test_pt_2_4, "Skipping test as OP support available from PyTorch 2.4"
)
class Test_Sdpa_Model(Zentorch_TestCase):
    @parameterized.expand(product(supported_dtypes, seq_length_opt, batch_size_opt))
    @torch.inference_mode()
    def test_sdpa_model(self, dtype, seq_length, batch_size):
        self.data.create_unittest_data(dtype)
        torch_type = self.data.get_torch_type(dtype)
        amp_enabled = True if dtype == "bfloat16" else False
        native_model = Custom_Model_Sdpa().eval()
        zentorch_model = Custom_Model_Sdpa().eval()
        zentorch_model = torch.compile(zentorch_model, backend="zentorch")
        with torch.inference_mode(), torch.autocast(
            device_type="cpu", enabled=amp_enabled
        ):
            sdpa_query = torch.randn(
                batch_size, 16, seq_length, 64, device="cpu", requires_grad=False
            ).type(torch_type)
            sdpa_key = torch.randn(
                batch_size, 16, seq_length, 64, device="cpu", requires_grad=False
            ).type(torch_type)
            sdpa_value = torch.randn(
                batch_size, 16, seq_length, 64, device="cpu", requires_grad=False
            ).type(torch_type)
            sdpa_attention_mask = None
            native_output = native_model(
                sdpa_query,
                sdpa_key,
                sdpa_value,
                sdpa_attention_mask,
            )
            zentorch_output = zentorch_model(
                sdpa_query,
                sdpa_key,
                sdpa_value,
                sdpa_attention_mask,
            )
            self.assertEqual(native_output, zentorch_output, atol=1e-3, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
