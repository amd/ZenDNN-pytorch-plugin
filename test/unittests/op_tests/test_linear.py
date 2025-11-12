# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    MMTestCase,
    has_zentorch,
    reset_dynamo,
    run_tests,
    skip_test_pt_2_0,
    supported_dtypes,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Linear_Unary(MMTestCase):
    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_bias(self):
        reset_dynamo()
        output_1 = torch.nn.functional.linear(
            self.data.x, self.data.y.t(), self.data.input1d
        )
        output_2 = torch.ops.zentorch.zentorch_linear_unary(
            self.data.x, self.data.y.t(), self.data.input1d
        )
        self.assertEqual(output_1, output_2, atol=1e-2, rtol=1e-2)

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_no_bias(self):
        reset_dynamo()
        output_1 = torch.nn.functional.linear(self.data.x, self.data.y.t())
        output_2 = torch.ops.zentorch.zentorch_linear_unary(
            self.data.x, self.data.y.t()
        )
        self.assertEqual(output_1, output_2, atol=1e-2, rtol=1e-2)

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_bias_relu(self):
        reset_dynamo()
        output_1 = torch.relu(
            torch.nn.functional.linear(self.data.x, self.data.y.t(), self.data.input1d)
        )
        output_2 = torch.ops.zentorch.zentorch_linear_unary(
            self.data.x, self.data.y.t(), self.data.input1d, post_op="relu"
        )
        self.assertEqual(output_1, output_2, atol=1e-2, rtol=1e-2)

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_no_bias_relu(self):
        reset_dynamo()
        output_1 = torch.relu(torch.nn.functional.linear(self.data.x, self.data.y.t()))
        output_2 = torch.ops.zentorch.zentorch_linear_unary(
            self.data.x, self.data.y.t(), post_op="relu"
        )
        self.assertEqual(output_1, output_2, atol=1e-2, rtol=1e-2)

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_bias_gelu_tanh(self):
        reset_dynamo()
        output_1 = torch.nn.GELU(approximate="tanh")(
            torch.nn.functional.linear(self.data.x, self.data.y.t(), self.data.input1d)
        )
        output_2 = torch.ops.zentorch.zentorch_linear_unary(
            self.data.x, self.data.y.t(), self.data.input1d, post_op="gelu_tanh"
        )
        self.assertEqual(output_1, output_2, atol=1e-2, rtol=1e-2)

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_no_bias_gelu_tanh(self):
        reset_dynamo()
        output_1 = torch.nn.GELU(approximate="tanh")(
            torch.nn.functional.linear(self.data.x, self.data.y.t())
        )
        output_2 = torch.ops.zentorch.zentorch_linear_unary(
            self.data.x, self.data.y.t(), post_op="gelu_tanh"
        )
        self.assertEqual(output_1, output_2, atol=1e-2, rtol=1e-2)

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_bias_gelu_erf(self):
        reset_dynamo()
        output_1 = torch.nn.GELU()(
            torch.nn.functional.linear(self.data.x, self.data.y.t(), self.data.input1d)
        )
        output_2 = torch.ops.zentorch.zentorch_linear_unary(
            self.data.x, self.data.y.t(), self.data.input1d, post_op="gelu_erf"
        )
        self.assertEqual(output_1, output_2, atol=1e-2, rtol=1e-2)

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_no_bias_gelu_erf(self):
        reset_dynamo()
        output_1 = torch.nn.GELU()(
            torch.nn.functional.linear(self.data.x, self.data.y.t())
        )
        output_2 = torch.ops.zentorch.zentorch_linear_unary(
            self.data.x, self.data.y.t(), post_op="gelu_erf"
        )
        self.assertEqual(output_1, output_2, atol=1e-2, rtol=1e-2)

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_bias_silu(self):
        reset_dynamo()
        output_1 = torch.nn.functional.silu(
            torch.nn.functional.linear(self.data.x, self.data.y.t(), self.data.input1d)
        )
        output_2 = torch.ops.zentorch.zentorch_linear_unary(
            self.data.x, self.data.y.t(), self.data.input1d, post_op="silu"
        )
        self.assertEqual(output_1, output_2, atol=1e-2, rtol=1e-2)

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_no_bias_silu(self):
        reset_dynamo()
        output_1 = torch.nn.functional.silu(
            torch.nn.functional.linear(self.data.x, self.data.y.t())
        )
        output_2 = torch.ops.zentorch.zentorch_linear_unary(
            self.data.x, self.data.y.t(), post_op="silu"
        )
        self.assertEqual(output_1, output_2, atol=1e-2, rtol=1e-2)

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_bias_sigmoid(self):
        reset_dynamo()
        output_1 = torch.nn.functional.sigmoid(
            torch.nn.functional.linear(self.data.x, self.data.y.t(), self.data.input1d)
        )
        output_2 = torch.ops.zentorch.zentorch_linear_unary(
            self.data.x, self.data.y.t(), self.data.input1d, post_op="sigmoid"
        )
        self.assertEqual(output_1, output_2, atol=1e-2, rtol=1e-2)

    @MMTestCase.hypothesis_params_mm_itr(dtype_list=supported_dtypes)
    @unittest.skipIf(skip_test_pt_2_0, "Skipping test due to PT2.0 instability")
    def test_linear_unary_with_no_bias_sigmoid(self):
        reset_dynamo()
        output_1 = torch.nn.functional.sigmoid(
            torch.nn.functional.linear(self.data.x, self.data.y.t())
        )
        output_2 = torch.ops.zentorch.zentorch_linear_unary(
            self.data.x, self.data.y.t(), post_op="sigmoid"
        )
        self.assertEqual(output_1, output_2, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
