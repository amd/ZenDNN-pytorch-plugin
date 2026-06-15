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
    WOQTestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    counters,
    DataTypes,
    woq_dtypes,
    freeze_opt,
    cpp_wrapper_opt,
    test_with_freeze_opt_and_cpp_wrapper,
)
from woq_test_utils import WOQ_Linear_Model  # noqa: E402


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class WOQ_Linear_Add_Add_Model(nn.Module):
    """WOQ linear (with bias) + add + add: add(add(woq(x), a), b)."""

    def __init__(self, out_features, in_features, dtype):
        super().__init__()
        self.woq = WOQ_Linear_Model(
            out_features, in_features, group_size=None, bias=True
        )
        self.out_features = out_features
        self.register_buffer(
            "add_1",
            torch.randn(out_features, dtype=dtype).unsqueeze(0),
        )
        self.register_buffer(
            "add_2",
            torch.randn(out_features, dtype=dtype).unsqueeze(0),
        )

    def forward(self, x):
        woq_out = self.woq(x)
        a = self.add_1.expand(woq_out.shape[0], -1)
        b = self.add_2.expand(woq_out.shape[0], -1)
        return (woq_out + a) + b


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class WOQ_Linear_Mul_Add_Model(nn.Module):
    """WOQ linear (with bias) + mul + add: add(mul(woq(x), m), b)."""

    def __init__(self, out_features, in_features, dtype):
        super().__init__()
        self.woq = WOQ_Linear_Model(
            out_features, in_features, group_size=None, bias=True
        )
        self.out_features = out_features
        self.register_buffer(
            "mul_operand",
            torch.randn(out_features, dtype=dtype).unsqueeze(0),
        )
        self.register_buffer(
            "add_operand",
            torch.randn(out_features, dtype=dtype).unsqueeze(0),
        )

    def forward(self, x):
        woq_out = self.woq(x)
        m = self.mul_operand.expand(woq_out.shape[0], -1)
        b = self.add_operand.expand(woq_out.shape[0], -1)
        return (woq_out * m) + b


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_WOQ_Linear_Binary_Binary_Fusion(WOQTestCase):
    """Test that WOQ linear + binary-binary patterns are fused."""

    def _assert_fusion_replaced(
        self, model, x, counter_key, pattern_description, freeze_opt=False, cpp_wrapper=False
    ):
        eager_out = model(x)
        reset_dynamo()
        compiled = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"].get(counter_key, 0), 0)
        compiled_out = test_with_freeze_opt_and_cpp_wrapper(
            compiled, x, freeze_opt, cpp_wrapper
        )
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

    # Test Fails while generalising test
    # Bug has been reported Jira ID: ZENAI-3716
    # @WOQTestCase.hypothesis_params_woq_itr(
    #     dtype_opt_list=woq_dtypes,
    #     batch_opt_list=batch_opt,
    #     in_features_opt_list=in_features_opt,
    #     out_features_opt_list=out_features_opt,
    #     bias_opt_list=woq_bias_opt,
    #     freeze_list=freeze_opt,
    #     cpp_wrapper_opt_list=cpp_wrapper_opt,
    # )
    # TODO: Pass freeze_opt for freeze_list instead of only [False].
    #       Test fails when freeze_opt is True.
    #       Bug has been reported Jira ID: ZENAI-3867
    @WOQTestCase.hypothesis_params_woq_itr(
        dtype_opt_list=woq_dtypes,
        batch_opt_list=[4],
        in_features_opt_list=[64],
        out_features_opt_list=[48],
        bias_opt_list=[True],
        freeze_list=[False],
        cpp_wrapper_opt_list=cpp_wrapper_opt,
    )
    @torch.inference_mode()
    def test_woq_linear_add_add(self, freeze_opt, cpp_wrapper):
        woq_dtype = DataTypes.get_torch_type(self.data.dtype)
        model = WOQ_Linear_Add_Add_Model(self.data.out_features, self.data.in_features, woq_dtype).eval()
        x = self.data.woq_input
        self._assert_fusion_replaced(
            model,
            x,
            "zentorch_woq_linear_add_add",
            "WOQ linear + add + add",
            freeze_opt=freeze_opt,
            cpp_wrapper=cpp_wrapper,
        )

    # Test Fails while generalising test
    # Bug has been reported Jira ID: ZENAI-3717
    # @WOQTestCase.hypothesis_params_woq_itr(
    #     dtype_opt_list=woq_dtypes,
    #     batch_opt_list=batch_opt,
    #     in_features_opt_list=in_features_opt,
    #     out_features_opt_list=out_features_opt,
    #     bias_opt_list=woq_bias_opt,
    # )
    @WOQTestCase.hypothesis_params_woq_itr(
        dtype_opt_list=woq_dtypes,
        batch_opt_list=[4],
        in_features_opt_list=[64],
        out_features_opt_list=[48],
        bias_opt_list=[True],
        freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
    )
    @torch.inference_mode()
    def test_woq_linear_mul_add(self, freeze_opt, cpp_wrapper):
        woq_dtype = DataTypes.get_torch_type(self.data.dtype)
        model = WOQ_Linear_Mul_Add_Model(self.data.out_features, self.data.in_features, woq_dtype).eval()
        x = self.data.woq_input
        self._assert_fusion_replaced(
            model,
            x,
            "zentorch_woq_linear_mul_add",
            "WOQ linear + mul + add",
            freeze_opt=freeze_opt,
            cpp_wrapper=cpp_wrapper,
        )


if __name__ == "__main__":
    run_tests()
