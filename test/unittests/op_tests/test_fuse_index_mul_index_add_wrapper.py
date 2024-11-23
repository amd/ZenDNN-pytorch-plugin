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
    skip_test_pt_2_4,
)

batch_sizes = [1, 16]
seq_lens = [1, 32, 128]
num_experts = [4, 8, 12, 16]
top_k = [2, 4]


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
@unittest.skipIf(
    skip_test_pt_2_4, "Skipping test as OP support available from PyTorch 2.4"
)
class Test_Fuse_Index_Mul_Index_Add_Wrapper(Zentorch_TestCase):
    @parameterized.expand(product(batch_sizes, seq_lens, num_experts, top_k))
    @torch.inference_mode()
    def test_fuse_index_mul_index_add_wrapper(
        self, batch_size, seq_len, block_sparse_moe_num_experts, block_sparse_moe_top_k
    ):
        hidden_dim = 32

        router_logits = torch.randn(
            (batch_size * seq_len, block_sparse_moe_num_experts), dtype=torch.bfloat16
        )
        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=1, dtype=torch.float
        )
        routing_weights, selected_experts = torch.topk(
            routing_weights, block_sparse_moe_top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(torch.bfloat16)

        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=block_sparse_moe_num_experts
        ).permute(2, 1, 0)
        expert_idx = torch.randint(
            low=0, high=block_sparse_moe_num_experts, size=(1,)
        ).item()
        idx, top_x = torch.where(expert_mask[expert_idx])

        # The initialization of curr_state is based on the model run and the
        # outputs are printed and checked.
        # Based on the outputs, the curr_state is initialized.
        curr_state = torch.randn(
            (int(idx.shape[0]), hidden_dim), dtype=torch.bfloat16
        ).unsqueeze(0)
        zentorch_output = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=torch.bfloat16,
        )

        zentorch_output = torch.ops.zentorch.fuse_index_mul_index_add(
            curr_state,
            top_x,
            idx,
            routing_weights,
            zentorch_output,
        )

        output_ref = torch.zeros(
            (batch_size * seq_len, hidden_dim), dtype=torch.bfloat16
        )
        routing_w = routing_weights[top_x, idx].unsqueeze(-1)
        curr_state = curr_state * routing_w
        output_ref.index_add_(0, top_x, curr_state.squeeze(0))

        self.assertEqual(zentorch_output, output_ref)

    @parameterized.expand(product(batch_sizes, seq_lens))
    @torch.inference_mode()
    def test_fuse_index_mul_index_add_wrapper_incorrect_dtype(
        self, batch_size, seq_len
    ):
        block_sparse_moe_num_experts = 8
        hidden_dim = 32
        block_sparse_moe_top_k = 2

        router_logits = torch.randn(
            (batch_size * seq_len, block_sparse_moe_num_experts), dtype=torch.bfloat16
        )
        routing_weights = torch.nn.functional.softmax(
            router_logits, dim=1, dtype=torch.float
        )
        routing_weights, selected_experts = torch.topk(
            routing_weights, block_sparse_moe_top_k, dim=-1
        )
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(torch.bfloat16)

        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=block_sparse_moe_num_experts
        ).permute(2, 1, 0)
        expert_idx = torch.randint(
            low=0, high=block_sparse_moe_num_experts, size=(1,)
        ).item()
        idx, top_x = torch.where(expert_mask[expert_idx])

        curr_state = torch.randn((1, int(idx.shape[0]), hidden_dim)).to(torch.int)
        zentorch_output = torch.zeros(
            (batch_size * seq_len, hidden_dim),
            dtype=torch.bfloat16,
        )

        with self.assertRaises(RuntimeError) as context:
            _ = torch.ops.zentorch.fuse_index_mul_index_add(
                curr_state,
                top_x,
                idx,
                routing_weights,
                zentorch_output,
            )
        self.assertTrue(
            "zentorch::fuse_index_mul_index_add supports only "
            "bfloat16 datatype" in str(context.exception)
        )


if __name__ == "__main__":
    run_tests()
