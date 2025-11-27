# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import operator
import unittest
import torch
from torch import nn
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    AddmmTestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    zentorch,
    freeze_opt,
    test_with_freeze_opt,
    counters,
)


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

    def setUp(self):
        super().setUp()
        self._original_zentorch_linear = os.environ.get("ZENTORCH_LINEAR")
        os.environ["ZENTORCH_LINEAR"] = "1"

    def tearDown(self):
        if self._original_zentorch_linear is None:
            os.environ.pop("ZENTORCH_LINEAR", None)
        else:
            os.environ["ZENTORCH_LINEAR"] = self._original_zentorch_linear
        super().tearDown()

    def _run_binary_post_op(
        self, key: str, bias_flag: bool, dtype: str, freeze_flag: bool
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
        binary_tensor = torch.randn_like(reference_linear)
        native_output = LINEAR_BINARY_OPS[key]["op"](reference_linear, binary_tensor)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        counters.clear()
        counter_key = LINEAR_BINARY_OPS[key]["counter"]
        self.assertEqual(counters["zentorch"][counter_key], 0)
        compiled_output = test_with_freeze_opt(
            compiled_graph,
            (self.data.input, binary_tensor),
            freeze_flag,
        )
        self.assertEqual(counters["zentorch"][counter_key], 1)
        tolerance = LINEAR_TOLERANCES.get(dtype, {"atol": 1e-3, "rtol": 1e-3})
        self.assertEqual(native_output, compiled_output, **tolerance)

    @AddmmTestCase.hypothesis_params_addmm_itr(dtype_list=supported_dtypes, freeze_list=freeze_opt)
    @torch.inference_mode()
    def test_linear_add_model(self, dtype, freeze_opt):
        for bias_name, bias_flag in LINEAR_BIAS_CASES.items():
            with self.subTest(bias=bias_name):
                self._run_binary_post_op("add", bias_flag, dtype, freeze_opt)

    @AddmmTestCase.hypothesis_params_addmm_itr(dtype_list=supported_dtypes, freeze_list=freeze_opt)
    @torch.inference_mode()
    def test_linear_mul_model(self, dtype, freeze_opt):
        for bias_name, bias_flag in LINEAR_BIAS_CASES.items():
            with self.subTest(bias=bias_name):
                self._run_binary_post_op("mul", bias_flag, dtype, freeze_opt)


if __name__ == "__main__":
    run_tests()
