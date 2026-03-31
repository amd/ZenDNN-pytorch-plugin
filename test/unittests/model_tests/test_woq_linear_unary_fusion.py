# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

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
class WOQ_Linear_Unary_Model(nn.Module):
    """WOQ linear followed by a unary op (GELU_ERF, GELU_TANH)."""

    def __init__(self, out_features, in_features, unary_fn, bias=False):
        super().__init__()
        self.woq = WOQ_Linear_Model(
            out_features, in_features, group_size=None, bias=bias
        )
        self.unary_fn = unary_fn

    def forward(self, x):
        out = self.woq(x)
        return self.unary_fn(out)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQ_Linear_Unary_Fusion(Zentorch_TestCase):
    """Test that WOQ linear + unary patterns are matched and replaced by fused ops."""

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
    def test_woq_linear_gelu_no_bias(self):
        batch, in_features, out_features = 4, 64, 48
        model = WOQ_Linear_Unary_Model(
            out_features, in_features, torch.nn.functional.gelu, bias=False
        ).eval()
        x = torch.randn(batch, in_features, dtype=torch.bfloat16)
        self._assert_fusion_replaced(
            model, x, "zentorch_woq_linear_gelu_erf", "WOQ linear + GELU_ERF (no bias)"
        )

    @torch.inference_mode()
    def test_woq_linear_gelu_with_bias(self):
        batch, in_features, out_features = 4, 64, 48
        model = WOQ_Linear_Unary_Model(
            out_features, in_features, torch.nn.functional.gelu, bias=True
        ).eval()
        x = torch.randn(batch, in_features, dtype=torch.bfloat16)
        self._assert_fusion_replaced(
            model, x, "zentorch_woq_linear_gelu_erf", "WOQ linear + GELU_ERF (with bias)"
        )


if __name__ == "__main__":
    run_tests()
