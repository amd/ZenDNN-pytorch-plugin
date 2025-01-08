# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from parameterized import parameterized
from itertools import product
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    supported_dtypes,
    Test_Data,
)


sliding_windows = [10, 40]
seq_lens = [1, 32]


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Prepare_4d_Casual_Attention_Mask(Zentorch_TestCase):
    @parameterized.expand(product(supported_dtypes, sliding_windows, seq_lens))
    @torch.inference_mode()
    def test_prepare_4d_causal_attention_mask(self, dtype, sliding_window, seq_len):
        torch_dtype = torch.float32 if dtype == "float32" else torch.bfloat16
        inputs_embeds = torch.rand((1, seq_len, 768), dtype=torch_dtype)
        finfo_min = torch.finfo(torch_dtype).min
        past_key_values_length = 0
        if seq_len == 1:
            past_key_values_length = 32
        attention_mask = torch.ones(
            (1, past_key_values_length + seq_len), dtype=torch_dtype
        )
        output = torch.ops.zentorch.prepare_4d_causal_attention_mask(
            attention_mask,
            inputs_embeds,
            past_key_values_length,
            torch.tensor(finfo_min).contiguous(),
            sliding_window,
        )

        from transformers.modeling_attn_mask_utils import (
            _prepare_4d_causal_attention_mask,
        )

        output_ref = _prepare_4d_causal_attention_mask(
            attention_mask,
            (inputs_embeds.shape[0], inputs_embeds.shape[1]),
            inputs_embeds,
            past_key_values_length,
            sliding_window,
        )
        self.assertEqual(output, output_ref)

    @parameterized.expand(product(sliding_windows, seq_lens))
    def test_prepare_4d_causal_attention_mask_incorrect_dtype(
        self, sliding_window, seq_len
    ):
        inputs_embeds = torch.randint(low=0, high=100, size=(1, seq_len, 768))
        finfo_min = torch.iinfo(torch.int).min
        past_key_values_length = 0
        if seq_len == 1:
            past_key_values_length = 32
        attention_mask = torch.ones(
            (1, past_key_values_length + seq_len), dtype=torch.long
        )

        with self.assertRaises(RuntimeError) as context:
            _ = torch.ops.zentorch.prepare_4d_causal_attention_mask(
                attention_mask,
                inputs_embeds,
                past_key_values_length,
                torch.tensor(finfo_min).contiguous(),
                sliding_window,
            )
        self.assertTrue(
            "zentorch::prepare_4d_causal_attention_mask_kernel_impl supports "
            "only float and bfloat16 datatypes" in str(context.exception)
        )


if __name__ == "__main__":
    run_tests()
