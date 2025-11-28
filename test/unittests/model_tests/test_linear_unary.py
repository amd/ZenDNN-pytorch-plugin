# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
import os
from pathlib import Path
import copy

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

LINEAR_ACTIVATIONS = {
    "relu": {
        "factory": nn.ReLU,
        "kwargs": {},
        "counter": "zentorch_linear_relu",
    },
    "gelu_tanh": {
        "factory": nn.GELU,
        "kwargs": {"approximate": "tanh"},
        "counter": "zentorch_linear_gelu_tanh",
    },
    "gelu_erf": {
        "factory": nn.GELU,
        "kwargs": {"approximate": "none"},
        "counter": "zentorch_linear_gelu_erf",
    },
    "silu": {
        "factory": nn.SiLU,
        "kwargs": {},
        "counter": "zentorch_linear_silu",
    },
    "sigmoid": {
        "factory": nn.Sigmoid,
        "kwargs": {},
        "counter": "zentorch_linear_sigmoid",
    },
    "tanh": {
        "factory": nn.Tanh,
        "kwargs": {},
        "counter": "zentorch_linear_tanh",
    },
}


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Linear_Unary_Model(AddmmTestCase):

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

    def _run_activation(self, key, dtype, freeze_flag):
        case = LINEAR_ACTIVATIONS[key]
        linear = nn.Linear(
            self.data.n,
            self.data.k,
            dtype=self.data.get_torch_type(dtype),
        )
        activation = case["factory"](**case["kwargs"])
        model = nn.Sequential(linear, activation)
        model_copy = copy.deepcopy(model)
        compiled_graph_copy = torch.compile(model_copy)
        native_output = compiled_graph_copy(self.data.input)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"][case["counter"]], 0)
        compiled_output = test_with_freeze_opt(
            compiled_graph,
            (self.data.input),
            freeze_flag,
        )
        self.assertEqual(counters["zentorch"][case["counter"]], 1)
        self.assertEqual(native_output, compiled_output, atol=1e-3, rtol=1e-5)

    @AddmmTestCase.hypothesis_params_addmm_itr(dtype_list=supported_dtypes, freeze_list=freeze_opt)
    @torch.inference_mode()
    def test_linear_relu_model(self, dtype, freeze_opt):
        self._run_activation("relu", dtype, freeze_opt)

    @AddmmTestCase.hypothesis_params_addmm_itr(dtype_list=supported_dtypes, freeze_list=freeze_opt)
    @torch.inference_mode()
    def test_linear_gelu_tanh_model(self, dtype, freeze_opt):
        self._run_activation("gelu_tanh", dtype, freeze_opt)

    @AddmmTestCase.hypothesis_params_addmm_itr(dtype_list=supported_dtypes, freeze_list=freeze_opt)
    @torch.inference_mode()
    def test_linear_gelu_erf_model(self, dtype, freeze_opt):
        self._run_activation("gelu_erf", dtype, freeze_opt)

    @AddmmTestCase.hypothesis_params_addmm_itr(dtype_list=supported_dtypes, freeze_list=freeze_opt)
    @torch.inference_mode()
    def test_linear_silu_model(self, dtype, freeze_opt):
        self._run_activation("silu", dtype, freeze_opt)

    @AddmmTestCase.hypothesis_params_addmm_itr(dtype_list=supported_dtypes, freeze_list=freeze_opt)
    @torch.inference_mode()
    def test_linear_sigmoid_model(self, dtype, freeze_opt):
        self._run_activation("sigmoid", dtype, freeze_opt)

    @AddmmTestCase.hypothesis_params_addmm_itr(dtype_list=supported_dtypes, freeze_list=freeze_opt)
    @torch.inference_mode()
    def test_linear_tanh_model(self, dtype, freeze_opt):
        self._run_activation("tanh", dtype, freeze_opt)


if __name__ == "__main__":
    run_tests()
