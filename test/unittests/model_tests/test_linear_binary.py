# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import operator
import unittest
import torch
from torch.testing import FileCheck
from torch import nn
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

LINEAR_BIAS_CASES = {
    "with_bias": True,
    "no_bias": False,
}

LINEAR_TOLERANCES = {
    "float32": {"atol": 1e-3, "rtol": 1e-3},
    "bfloat16": {"atol": 5e-2, "rtol": 5e-2},
}


class LinearBinaryModel(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, dtype: torch.dtype, op, bias: bool
    ):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias, dtype=dtype)
        self.binary_op = op

    def forward(
        self, input_tensor: torch.Tensor, binary_tensor: torch.Tensor
    ) -> torch.Tensor:
        return self.binary_op(self.linear(input_tensor), binary_tensor)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Linear_Binary_Model(AddmmTestCase):

    def _run_binary_post_op(
        self,
        key: str,
        bias_flag: bool,
        dtype: str,
        freeze_flag: bool,
        cpp_wrapper: bool = False,
        transposed_binary: bool = False,
    ) -> None:
        if dtype == "bfloat16":
            self.skip_if_bfloat16_unsupported_hardware()
        torch_dtype = self.data.get_torch_type(dtype)
        model = LinearBinaryModel(
            self.data.n,
            self.data.m,
            torch_dtype,
            LINEAR_BINARY_OPS[key]["op"],
            bias=bias_flag,
        )
        reference_linear = model.linear(self.data.input)
        if transposed_binary:
            binary_tensor = torch.randn(
                reference_linear.shape[::-1], dtype=torch_dtype
            ).t()
        else:
            binary_tensor = torch.randn_like(reference_linear)
        native_output = LINEAR_BINARY_OPS[key]["op"](reference_linear, binary_tensor)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        counters.clear()
        counter_key = LINEAR_BINARY_OPS[key]["counter"]
        self.assertEqual(counters["zentorch"][counter_key], 0)
        compiled_output, cpp_code = test_with_freeze_opt_and_cpp_wrapper(
            compiled_graph,
            (self.data.input, binary_tensor),
            freeze_flag,
            cpp_wrapper,
        )
        if transposed_binary:
            # We don't expect fusion for transposed binary
            self.assertEqual(counters["zentorch"][counter_key], 0)
        else:
            self.assertEqual(counters["zentorch"][counter_key], 1)
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
        dtype_list=supported_dtypes, freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
        # cold cpp_wrapper compile exceeds the default deadline; see
        # test_with_freeze_opt_and_cpp_wrapper in zentorch_test_utils.
        time_out=60000,
    )
    @torch.inference_mode()
    def test_linear_add_model(self, dtype, freeze_opt, cpp_wrapper):
        for bias_name, bias_flag in LINEAR_BIAS_CASES.items():
            with self.subTest(bias=bias_name):
                self._run_binary_post_op("add", bias_flag, dtype, freeze_opt, cpp_wrapper)

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes, freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
        # cold cpp_wrapper compile exceeds the default deadline; see
        # test_with_freeze_opt_and_cpp_wrapper in zentorch_test_utils.
        time_out=60000,
    )
    @torch.inference_mode()
    def test_linear_mul_model(self, dtype, freeze_opt, cpp_wrapper):
        for bias_name, bias_flag in LINEAR_BIAS_CASES.items():
            with self.subTest(bias=bias_name):
                self._run_binary_post_op("mul", bias_flag, dtype, freeze_opt, cpp_wrapper)

    @AddmmTestCase.hypothesis_params_addmm_itr(
        dtype_list=supported_dtypes, freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
        # cold cpp_wrapper compile exceeds the default deadline; see
        # test_with_freeze_opt_and_cpp_wrapper in zentorch_test_utils.
        time_out=60000,
    )
    @torch.inference_mode()
    def test_linear_add_model_with_transposed_binary(self, dtype, freeze_opt, cpp_wrapper):
        for bias_name, bias_flag in LINEAR_BIAS_CASES.items():
            with self.subTest(bias=bias_name):
                self._run_binary_post_op(
                    "add", bias_flag, dtype, freeze_opt, cpp_wrapper, transposed_binary=True
                )


if __name__ == "__main__":
    run_tests()
