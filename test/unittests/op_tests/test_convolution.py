# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from itertools import product
from parameterized import parameterized
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    supported_dtypes,
    conv_stride,
    conv_padding,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Convolution(Zentorch_TestCase):
    @parameterized.expand([("int",)])
    def test_convolution_unsupported_dtype(self, dtype):
        self.data.create_unittest_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_convolution(
                self.data.conv_input,
                self.data.conv_weight,
                self.data.conv_bias,
                self.data.stride,
                self.data.padding,
                self.data.dilation,
                False,
                self.data.output_padding,
                1,
            )
        self.assertTrue(
            "unsupported data type, only bf16 and fp32 supported for now"
            in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    def test_convolution_invalid_dims(self, dtype):
        self.data.create_unittest_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_convolution(
                self.data.conv_input3d,
                self.data.conv_weight3d,
                self.data.conv_bias,
                self.data.stride,
                self.data.padding,
                self.data.dilation,
                False,
                self.data.output_padding,
                1,
            )
        self.assertTrue(
            "unsupported dims for conv input and weight" in str(context.exception)
        )

    @parameterized.expand(supported_dtypes)
    def test_convolution_unsupported_dilation(self, dtype):
        self.data.create_unittest_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_convolution(
                self.data.conv_input,
                self.data.conv_weight,
                self.data.conv_bias,
                self.data.stride,
                self.data.padding,
                self.data.dilation2,
                False,
                self.data.output_padding,
                1,
            )
        self.assertTrue(
            "unsupported value of dilation, only [1,1] supported for now"
            in str(context.exception)
        )

    @parameterized.expand(
        product(
            supported_dtypes,
            conv_stride,
            conv_padding,
        )
    )
    def test_convolution(self, dtype, stride, padding):
        self.data.create_unittest_data(dtype)
        conv_output = torch._C._VariableFunctions.convolution(
            self.data.conv_input,
            self.data.conv_weight,
            self.data.conv_bias,
            stride,
            padding,
            self.data.dilation,
            False,
            self.data.output_padding,
            1,
        )

        conv_output_z = torch.ops.zentorch.zentorch_convolution(
            self.data.conv_input,
            self.data.conv_weight,
            self.data.conv_bias,
            stride,
            padding,
            self.data.dilation,
            False,
            self.data.output_padding,
            1,
        )
        self.assertEqual(conv_output, conv_output_z)


if __name__ == "__main__":
    run_tests()
