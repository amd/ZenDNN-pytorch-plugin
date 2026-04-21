# ******************************************************************************
# Copyright (c) 2024-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
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


class Custom_Deep_Linear_Activation_Model(nn.Module):
    """
    Model with three linear layers + activation, intermediate activation,
    then three more linear layers + activation
    """
    def __init__(self, k, dtype):
        super().__init__()
        self.post_op = torch.nn.ReLU()
        self.bmlp_0 = nn.Linear(k, 512, dtype=dtype)
        self.bmlp_1 = nn.Linear(512, 256, dtype=dtype)
        self.bmlp_2 = nn.Linear(256, 64, dtype=dtype)
        self.intermediate_activation = torch.nn.Sigmoid()
        self.tmlp_0 = nn.Linear(64, 32, dtype=dtype)
        self.tmlp_1 = nn.Linear(32, 16, dtype=dtype)
        self.tmlp_2 = nn.Linear(16, 8, dtype=dtype)

    def forward(self, inputs):
        outputs = self.bmlp_0(inputs)
        outputs = self.post_op(outputs)
        outputs = self.bmlp_1(outputs)
        outputs = self.post_op(outputs)
        outputs = self.bmlp_2(outputs)
        outputs = self.post_op(outputs)
        outputs = self.intermediate_activation(outputs)
        outputs = self.tmlp_0(outputs)
        outputs = self.post_op(outputs)
        outputs = self.tmlp_1(outputs)
        outputs = self.post_op(outputs)
        outputs = self.tmlp_2(outputs)
        outputs = self.post_op(outputs)
        return outputs


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Linear_Unary_Model(AddmmTestCase):

    def _run_activation(self, key, dtype, freeze_flag, model=None, input_tensor=None, expected_count=1):
        case = LINEAR_ACTIVATIONS[key]
        if model is None:
            linear = nn.Linear(
                self.data.n,
                self.data.k,
                dtype=self.data.get_torch_type(dtype),
            )
            activation = case["factory"](**case["kwargs"])
            model = nn.Sequential(linear, activation)
            input_tensor = self.data.input
        if input_tensor is None:
            input_tensor = self.data.input

        model_copy = copy.deepcopy(model)
        compiled_graph_copy = torch.compile(model_copy)
        native_output = compiled_graph_copy(input_tensor)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        counters.clear()
        self.assertEqual(counters["zentorch"][case["counter"]], 0)
        compiled_output = test_with_freeze_opt(
            compiled_graph,
            (input_tensor,),
            freeze_flag,
        )
        self.assertEqual(counters["zentorch"][case["counter"]], expected_count)
        self.assertEqual(native_output, compiled_output, atol=1e-3, rtol=1e-5)
        # also compute for cpp_wrapper is freeze_flag is True
        # TODO: modify the test_with_freeze_opt to support cpp_wrapper (big change), and add other tests
        if freeze_flag:
            torch._inductor.config.cpp_wrapper = True
            native_output_cpp_wrapper = compiled_graph_copy(input_tensor)
            compiled_output_cpp_wrapper = test_with_freeze_opt(
                compiled_graph,
                (input_tensor,),
                freeze_flag,
            )
            self.assertEqual(native_output_cpp_wrapper, compiled_output_cpp_wrapper)
            torch._inductor.config.cpp_wrapper = False

    def _run_deep_linear_activation(self, key, dtype, freeze_flag):
        model = Custom_Deep_Linear_Activation_Model(
            self.data.k,
            dtype=self.data.get_torch_type(dtype),
        )
        self._run_activation(
            key,
            dtype,
            freeze_flag,
            model=model,
            input_tensor=self.data.x,
            expected_count=6,
        )

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

    @AddmmTestCase.hypothesis_params_addmm_itr(dtype_list=supported_dtypes, freeze_list=freeze_opt)
    @torch.inference_mode()
    def test_deep_linear_relu_sigmoid_model(self, dtype, freeze_opt):
        self._run_deep_linear_activation("relu", dtype, freeze_opt)


if __name__ == "__main__":
    run_tests()
