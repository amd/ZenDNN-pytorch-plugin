# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

"""Test that WOQ linear + binary-binary (add-add, mul-add) patterns are fused."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import copy  # noqa: E402
import unittest  # noqa: E402
import torch  # noqa: E402
from torch.testing import FileCheck  # noqa: E402
from torch import nn  # noqa: E402
from torch._inductor import config as inductor_config  # noqa: E402

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
    batch_opt,
    in_features_opt,
    out_features_opt,
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
        self, model, x, counter_key, pattern_description, freeze_opt=False,
        cpp_wrapper=False, check_out_variant=False
    ):
        # Reference: compile a clone of the model with the inductor backend.
        reset_dynamo()
        inductor_graph = torch.compile(copy.deepcopy(model), backend="inductor")
        inductor_out = inductor_graph(x)

        reset_dynamo()
        compiled = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"].get(counter_key, 0), 0)
        if check_out_variant:
            self.assertEqual(counters["zentorch"].get(f"{counter_key}_out", 0), 0)
        compiled_out, cpp_code = test_with_freeze_opt_and_cpp_wrapper(
            compiled, x, freeze_opt, cpp_wrapper
        )
        self.assertEqual(
            counters["zentorch"][counter_key],
            1,
            f"{pattern_description} should be replaced by exactly one "
            f"{counter_key}",
        )
        if check_out_variant:
            self.assertEqual(
                counters["zentorch"][f"{counter_key}_out"],
                1,
                f"{pattern_description} should lower to the {counter_key} out variant",
            )
        self.assertEqual(compiled_out.dtype, inductor_out.dtype)
        self.assertTrue(
            torch.allclose(compiled_out, inductor_out, rtol=1e-2, atol=1e-2),
            f"Compiled {pattern_description} output should match the "
            f"inductor-compiled reference.",
        )
        # Pillar 2 (codegen): op lowers to its AOTI out C-shim (see docstring).
        if cpp_wrapper:
            FileCheck().check(f"aoti_torch_cpu_{counter_key}_out").run(cpp_code)

        # Pinned bias_opt_list to [True]: the binary-binary fusion has no bias check, so bias=False
        # falls back to plain zentorch_woq_linear and skips the fusion under test.
        # The bias check was not added since the bias=False pattern was not
        # observed in any models.
    @WOQTestCase.hypothesis_params_woq_itr(
        dtype_opt_list=woq_dtypes,
        batch_opt_list=batch_opt,
        in_features_opt_list=in_features_opt,
        out_features_opt_list=out_features_opt,
        bias_opt_list=[True],
        freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
        # cold cpp_wrapper compile exceeds the default deadline; see
        # test_with_freeze_opt_and_cpp_wrapper in zentorch_test_utils.
        time_out=60000,
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

    # Caches off: the `.out` lowering counter is skipped on an FxGraphCache hit.
    # Smallest sweep, so it carries the counter check for the binary-binary family.
    @inductor_config.patch(force_disable_caches=True)
    @WOQTestCase.hypothesis_params_woq_itr(
        dtype_opt_list=woq_dtypes,
        batch_opt_list=batch_opt,
        in_features_opt_list=in_features_opt,
        out_features_opt_list=out_features_opt,
        bias_opt_list=[True],
        freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
        # cold cpp_wrapper compile exceeds the default deadline; see
        # test_with_freeze_opt_and_cpp_wrapper in zentorch_test_utils.
        time_out=60000,
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
            check_out_variant=True,
        )


if __name__ == "__main__":
    run_tests()
