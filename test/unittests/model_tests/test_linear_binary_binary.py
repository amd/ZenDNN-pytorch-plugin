# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import operator
import unittest
import torch
from torch.testing import FileCheck
from torch import nn
from torch._inductor import config as inductor_config
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    AddmmTestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    update_supported_dtypes,
    zentorch,
    freeze_opt,
    cpp_wrapper_opt,
    test_with_freeze_opt_and_cpp_wrapper,
    counters,
    bias_opt,
)

supported_dtypes = update_supported_dtypes(supported_dtypes, "zentorch_linear")

LINEAR_BINARY_OPS = {
    "add": {
        "op": operator.add,
        "counter": "zentorch_linear_add",
    },
    "mul": {
        "op": operator.mul,
        "counter": "zentorch_linear_mul",
    },
}

# Define combinations of two binary operations
LINEAR_BINARY_BINARY_OPS = {
    "add_add": {
        "op1": operator.add,
        "op2": operator.add,
        "counter": "zentorch_linear_add_add",
    },
    "mul_add": {
        "op1": operator.mul,
        "op2": operator.add,
        "counter": "zentorch_linear_mul_add",
    },
}

LINEAR_TOLERANCES = {
    "float32": {"atol": 1e-3, "rtol": 1e-3},
    "bfloat16": {"atol": 5e-2, "rtol": 5e-2},
}


class LinearBinaryBinaryModel(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        dtype: torch.dtype,
        op1,
        op2,
        bias: bool,
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        self.binary_op1 = op1
        self.binary_op2 = op2

    def forward(
        self,
        input_tensor: torch.Tensor,
        binary_tensor1: torch.Tensor,
        binary_tensor2: torch.Tensor,
    ) -> torch.Tensor:
        linear_out = self.linear(input_tensor)
        first_binary_out = self.binary_op1(linear_out, binary_tensor1)
        return self.binary_op2(first_binary_out, binary_tensor2)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Linear_Binary_Binary_Model(AddmmTestCase):

    def _run_binary_binary_post_op(
        self, key: str, bias_flag: bool, dtype: str, freeze_flag: bool,
        cpp_wrapper: bool = False, check_out_variant: bool = False
    ) -> None:
        if dtype == "bfloat16":
            self.skip_if_bfloat16_unsupported_hardware()
        torch_dtype = self.data.get_torch_type(dtype)
        model = LinearBinaryBinaryModel(
            self.data.n,
            self.data.m,
            torch_dtype,
            LINEAR_BINARY_BINARY_OPS[key]["op1"],
            LINEAR_BINARY_BINARY_OPS[key]["op2"],
            bias=bias_flag,
        )
        reference_linear = model.linear(self.data.input)
        binary_tensor1 = torch.randn_like(reference_linear)
        binary_tensor2 = torch.randn_like(reference_linear)

        # Compute native output
        first_binary_out = LINEAR_BINARY_BINARY_OPS[key]["op1"](
            reference_linear, binary_tensor1
        )
        native_output = LINEAR_BINARY_BINARY_OPS[key]["op2"](
            first_binary_out, binary_tensor2
        )

        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        counters.clear()
        counter_key = LINEAR_BINARY_BINARY_OPS[key]["counter"]
        self.assertEqual(counters["zentorch"][counter_key], 0)
        if check_out_variant:
            self.assertEqual(
                counters["zentorch"]["zentorch_linear_binary_binary_out"], 0
            )
        compiled_output, cpp_code = test_with_freeze_opt_and_cpp_wrapper(
            compiled_graph,
            (self.data.input, binary_tensor1, binary_tensor2),
            freeze_flag,
            cpp_wrapper,
        )
        self.assertEqual(counters["zentorch"][counter_key], 1)
        if check_out_variant:
            self.assertEqual(
                counters["zentorch"]["zentorch_linear_binary_binary_out"], 1
            )
        if freeze_flag:
            self.assertEqual(
                counters["zentorch"]["zentorch_weight_prepack_for_linear"], 1
            )
        tolerance = LINEAR_TOLERANCES.get(dtype, {"atol": 1e-3, "rtol": 1e-3})
        self.assertEqual(native_output, compiled_output, **tolerance)
        # Pillar 2 (codegen): op lowers to its AOTI C-shim (see helper docstring).
        if cpp_wrapper:
            FileCheck().check("aoti_torch_cpu_zentorch").run(cpp_code)

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes,
        freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
        bias_opt_list=bias_opt,
        # cold cpp_wrapper compile exceeds the default deadline; see
        # test_with_freeze_opt_and_cpp_wrapper in zentorch_test_utils.
        time_out=60000,
    )
    @torch.inference_mode()
    def test_linear_add_add_model(self, dtype, freeze_opt, cpp_wrapper, bias):
        self._run_binary_binary_post_op("add_add", bias, dtype, freeze_opt, cpp_wrapper)

    # The mul_add case also asserts the linear was lowered to its `.out`
    # variant (via check_out_variant). Caching is disabled since the lowering
    # counter is short-circuited on an FxGraphCache hit.
    @inductor_config.patch(force_disable_caches=True)
    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes,
        freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
        bias_opt_list=bias_opt,
        # cold cpp_wrapper compile exceeds the default deadline; see
        # test_with_freeze_opt_and_cpp_wrapper in zentorch_test_utils.
        time_out=60000,
    )
    @torch.inference_mode()
    def test_linear_mul_add_model(self, dtype, freeze_opt, cpp_wrapper, bias):
        self._run_binary_binary_post_op(
            "mul_add", bias, dtype, freeze_opt, cpp_wrapper,
            check_out_variant=True,
        )


if __name__ == "__main__":
    run_tests()
