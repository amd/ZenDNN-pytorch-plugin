# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from parameterized import parameterized
from itertools import product
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    supported_dtypes,
    zentorch,
    freeze_opt,
    test_with_freeze_opt,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Addmm_1dbias(nn.Module):
    def __init__(self, k, dtype) -> None:
        super(Custom_Model_Addmm_1dbias, self).__init__()
        self.mlp_0 = torch.nn.Linear(k, 512, dtype=dtype)
        self.mlp_1 = torch.nn.Linear(512, 256, dtype=dtype)
        self.mlp_2 = torch.nn.Linear(256, 64, dtype=dtype)

    def forward(self, inputs):
        outputs = self.mlp_0(inputs)
        outputs = self.mlp_1(outputs)
        outputs = self.mlp_2(outputs)

        return outputs


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Addmm_1dbias_Relu(nn.Module):
    def __init__(self, k, dtype) -> None:
        super(Custom_Model_Addmm_1dbias_Relu, self).__init__()

        self.post_op = torch.nn.ReLU()

        self.bmlp_0 = torch.nn.Linear(k, 512, dtype=dtype)
        self.bmlp_1 = torch.nn.Linear(512, 256, dtype=dtype)
        self.bmlp_2 = torch.nn.Linear(256, 64, dtype=dtype)

        self.intermediate_activation = torch.nn.Sigmoid()

        self.tmlp_0 = torch.nn.Linear(64, 32, dtype=dtype)
        self.tmlp_1 = torch.nn.Linear(32, 16, dtype=dtype)
        self.tmlp_2 = torch.nn.Linear(16, 8, dtype=dtype)

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
class Custom_Model_Addmm_1dbias_Relu_Gelu(nn.Module):
    def __init__(self, k, dtype) -> None:
        super(Custom_Model_Addmm_1dbias_Relu_Gelu, self).__init__()

        self.post_op_1 = torch.nn.ReLU()
        self.post_op_2 = torch.nn.GELU()

        self.bmlp_0 = torch.nn.Linear(k, 512, dtype=dtype)
        self.bmlp_1 = torch.nn.Linear(512, 256, dtype=dtype)
        self.bmlp_2 = torch.nn.Linear(256, 64, dtype=dtype)

        self.intermediate_activation = torch.nn.Sigmoid()

        self.tmlp_0 = torch.nn.Linear(64, 32, dtype=dtype)
        self.tmlp_1 = torch.nn.Linear(32, 16, dtype=dtype)
        self.tmlp_2 = torch.nn.Linear(16, 8, dtype=dtype)

    def forward(self, inputs):
        outputs = self.bmlp_0(inputs)
        outputs = self.post_op_1(outputs)
        outputs = self.bmlp_1(outputs)
        outputs = self.post_op_2(outputs)
        outputs = self.bmlp_2(outputs)
        outputs = self.post_op_1(outputs)

        outputs = self.intermediate_activation(outputs)

        outputs = self.tmlp_0(outputs)
        outputs = self.post_op_2(outputs)
        outputs = self.tmlp_1(outputs)
        outputs = self.post_op_1(outputs)
        outputs = self.tmlp_2(outputs)
        outputs = self.post_op_2(outputs)

        return outputs


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Addmm_1dbias_Model(Zentorch_TestCase):
    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_addmm_1dbias_model(self, dtype, freeze_opt):

        self.data.create_data(dtype)

        model = Custom_Model_Addmm_1dbias(self.data.k, self.data.get_torch_type(dtype))

        native_output = model(self.data.x)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(
            compiled_graph,
            (self.data.x),
            freeze_opt
        )
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_addmm_1dbias_relu_model(self, dtype, freeze_opt):

        self.data.create_data(dtype)

        model = Custom_Model_Addmm_1dbias_Relu(
            self.data.k, self.data.get_torch_type(dtype)
        )

        native_output = model(self.data.x)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(
            compiled_graph,
            (self.data.x),
            freeze_opt
        )
        self.assertEqual(native_output, compiled_output)

    @parameterized.expand(product(supported_dtypes, freeze_opt))
    @torch.inference_mode()
    def test_addmm_1dbias_relu_gelu_model(self, dtype, freeze_opt):

        self.data.create_data(dtype)

        model = Custom_Model_Addmm_1dbias_Relu_Gelu(
            self.data.k, self.data.get_torch_type(dtype)
        )

        native_output = model(self.data.x)
        reset_dynamo()
        compiled_graph = torch.compile(model, backend="zentorch")
        compiled_output = test_with_freeze_opt(
            compiled_graph,
            (self.data.x),
            freeze_opt
        )
        self.assertEqual(native_output, compiled_output, atol=1e-3, rtol=1e-5)


if __name__ == "__main__":
    run_tests()
