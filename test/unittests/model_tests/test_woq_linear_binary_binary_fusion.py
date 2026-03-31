# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Test that WOQ linear + binary-binary (add-add, mul-add) patterns are fused."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import unittest  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from unittest_utils import (  # noqa: E402
    has_zentorch,
    reset_dynamo,
    run_tests,
    counters,
    Zentorch_TestCase,
)
from woq_test_utils import WOQ_Linear_Model  # noqa: E402


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class WOQ_Linear_Add_Add_Model(nn.Module):
    """WOQ linear (with bias) + add + add: add(add(woq(x), a), b)."""

    def __init__(self, out_features, in_features):
        super().__init__()
        self.woq = WOQ_Linear_Model(
            out_features, in_features, group_size=None, bias=True
        )
        self.out_features = out_features
        self.register_buffer(
            "add_1",
            torch.randn(out_features, dtype=torch.bfloat16).unsqueeze(0),
        )
        self.register_buffer(
            "add_2",
            torch.randn(out_features, dtype=torch.bfloat16).unsqueeze(0),
        )

    def forward(self, x):
        woq_out = self.woq(x)
        a = self.add_1.expand(woq_out.shape[0], -1)
        b = self.add_2.expand(woq_out.shape[0], -1)
        return (woq_out + a) + b


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class WOQ_Linear_Mul_Add_Model(nn.Module):
    """WOQ linear (with bias) + mul + add: add(mul(woq(x), m), b)."""

    def __init__(self, out_features, in_features):
        super().__init__()
        self.woq = WOQ_Linear_Model(
            out_features, in_features, group_size=None, bias=True
        )
        self.out_features = out_features
        self.register_buffer(
            "mul_operand",
            torch.randn(out_features, dtype=torch.bfloat16).unsqueeze(0),
        )
        self.register_buffer(
            "add_operand",
            torch.randn(out_features, dtype=torch.bfloat16).unsqueeze(0),
        )

    def forward(self, x):
        woq_out = self.woq(x)
        m = self.mul_operand.expand(woq_out.shape[0], -1)
        b = self.add_operand.expand(woq_out.shape[0], -1)
        return (woq_out * m) + b


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQ_Linear_Binary_Binary_Fusion(Zentorch_TestCase):
    """Test that WOQ linear + binary-binary patterns are fused."""

    def _assert_fusion_replaced(self, model, x, counter_key, pattern_description):
        eager_out = model(x)
        reset_dynamo()
        compiled = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"].get(counter_key, 0), 0)
        compiled_out = compiled(x)
        self.assertEqual(
            counters["zentorch"][counter_key],
            1,
            f"{pattern_description} should be replaced by exactly one "
            f"{counter_key}",
        )
        self.assertTrue(
            torch.allclose(compiled_out, eager_out, rtol=1e-2, atol=1e-2),
            f"Compiled {pattern_description} output should match eager.",
        )

    @torch.inference_mode()
    def test_woq_linear_add_add(self):
        batch, in_features, out_features = 4, 64, 48
        model = WOQ_Linear_Add_Add_Model(out_features, in_features).eval()
        x = torch.randn(batch, in_features, dtype=torch.bfloat16)
        self._assert_fusion_replaced(
            model,
            x,
            "zentorch_woq_linear_add_add",
            "WOQ linear + add + add",
        )

    @torch.inference_mode()
    def test_woq_linear_mul_add(self):
        batch, in_features, out_features = 4, 64, 48
        model = WOQ_Linear_Mul_Add_Model(out_features, in_features).eval()
        x = torch.randn(batch, in_features, dtype=torch.bfloat16)
        self._assert_fusion_replaced(
            model,
            x,
            "zentorch_woq_linear_mul_add",
            "WOQ linear + mul + add",
        )


if __name__ == "__main__":
    run_tests()
