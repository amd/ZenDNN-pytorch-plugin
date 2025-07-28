# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import copy
import torch
import zentorch
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Shared_Weights(torch.nn.Module):
    def __init__(self, vocab=64, d_model=16):
        super().__init__()
        # one Parameter object shared by both layers
        shared = torch.nn.Parameter(torch.randn(vocab, d_model).to(torch.bfloat16))

        self.embed = torch.nn.Embedding(vocab, d_model, _weight=shared)
        self.proj = torch.nn.Linear(d_model, vocab, bias=False)
        self.proj.weight = shared  # <- tie

    def forward(self, tokens):
        x = self.embed(tokens)
        hid = x.mean(dim=1)
        # ── graph-break op ──
        _ = torch.nonzero(hid)  # dynamic -> Inductor can't infer len

        # ── subgraph ② (will compile) ──
        return self.proj(hid)  # uses the *same* weight again

    def get_shared_weight(self):
        return self.proj.weight


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Linear_Shared_Weights(torch.nn.Module):
    """
    Custom model with two different linear layers sharing the same weight
    with operations between them in the forward pass.
    """

    def __init__(self, input_size=64, hidden_size=64, output_size=32):
        super().__init__()

        # Create a shared weight parameter
        shared_weight = torch.nn.Parameter(
            torch.randn(hidden_size, input_size).to(torch.bfloat16)
        )

        # Create two linear layers that will share the same weight
        self.linear1 = torch.nn.Linear(input_size, hidden_size, bias=True)
        self.linear2 = torch.nn.Linear(input_size, hidden_size, bias=False)

        # Share the weight between the two linear layers
        self.linear1.weight = shared_weight
        self.linear2.weight = shared_weight  # <- weight sharing

        # Different biases for linear1 only (linear2 has bias=False)
        self.linear1.bias = torch.nn.Parameter(
            torch.randn(hidden_size).to(torch.bfloat16)
        )

        # Additional layers for operations between shared weight usages
        self.norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, x):
        # First linear transformation with shared weight
        h1 = self.linear1(x)

        # Operations between the shared weight usages
        h1_normalized = self.norm(h1)
        h1_scaled = h1_normalized * 0.5  # Scaling

        # Second linear transformation with the same shared weight
        # but applied to the processed input
        h2 = self.linear2(h1_scaled)

        return h2

    def get_shared_weight(self):
        """Return the shared weight for testing purposes."""
        return self.linear1.weight


# Fix for Milan unittest failure
# The below test is to check that shared parameters are handled
# correctly with MATMUL_ALGO=BF16:1
@unittest.skipIf(not zentorch._C.is_avx512_supported(), "No bf16 support on hardware")
@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Shared_Weights(Zentorch_TestCase):
    def setUp(self):
        # Store the original environment variable
        self.original_weight_caching = os.environ.get("ZENDNN_WEIGHT_CACHING", None)
        self.original_matmul_algo = os.environ.get("ZENDNN_MATMUL_ALGO", None)
        # Set the environment variable to 2
        os.environ["ZENDNN_WEIGHT_CACHING"] = "2"
        os.environ["ZENDNN_MATMUL_ALGO"] = "BF16:1"

    def tearDown(self):
        # Restore the original environment variable
        if self.original_weight_caching is not None:
            os.environ["ZENDNN_WEIGHT_CACHING"] = self.original_weight_caching
        else:
            del os.environ["ZENDNN_WEIGHT_CACHING"]

        if self.original_matmul_algo is not None:
            os.environ["ZENDNN_MATMUL_ALGO"] = self.original_matmul_algo
        else:
            del os.environ["ZENDNN_MATMUL_ALGO"]

    @torch.inference_mode()
    def test_shared_weights(self):
        model = Custom_Model_Shared_Weights()
        model_copy = copy.deepcopy(model)
        orig_weight = model.get_shared_weight().clone()

        model.eval()
        model_copy.eval()
        inp = torch.randint(0, 64, (4, 10), dtype=torch.long)

        model_copy = torch.compile(model_copy)
        with torch.no_grad():
            ref_out = model_copy(inp)

        model = zentorch.llm.optimize(model)
        model = torch.compile(model, backend="zentorch")
        with torch.no_grad():
            zentorch_out = model(inp)

        self.assertEqual(orig_weight, model.get_shared_weight())
        self.assertEqual(ref_out, zentorch_out)

    @torch.inference_mode()
    def test_linear_shared_weights(self):
        """Test the linear shared weights model with operations between shared usages."""
        model = Custom_Model_Linear_Shared_Weights()
        model_copy = copy.deepcopy(model)

        model.eval()
        model_copy.eval()
        inp = torch.randn(4, 64, dtype=torch.bfloat16)

        model_copy = torch.compile(model_copy)
        with torch.no_grad():
            ref_out = model_copy(inp)

        model = zentorch.llm.optimize(model)
        model = torch.compile(model, backend="zentorch")
        with torch.no_grad():
            zentorch_out = model(inp)
        self.assertEqual(ref_out, zentorch_out)


if __name__ == "__main__":
    run_tests()
