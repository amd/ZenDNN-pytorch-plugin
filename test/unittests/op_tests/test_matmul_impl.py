# ******************************************************************************
# Copyright (c) 2024-2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from parameterized import parameterized
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    run_tests,
    supported_dtypes,
    zentorch,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Matmul_Impl_Op(Zentorch_TestCase):
    @parameterized.expand(supported_dtypes)
    def test_matmul_impl_for_mv_and_dot(self, dtype):

        self.data.create_unittest_data(dtype)
        # mv
        self.assertEqual(
            torch.mv(self.data.input, self.data.input1d),
            zentorch._C.zentorch_matmul_impl(
                self.data.input,
                self.data.input1d,
                self.data.empty_bias,
                self.data.result_m,
                [],
                [],
            ),
            atol=1e-3,
            rtol=1e-2,
        )
        # dot
        self.assertEqual(
            torch.dot(self.data.input1d, self.data.input1d),
            zentorch._C.zentorch_matmul_impl(
                self.data.input1d,
                self.data.input1d,
                self.data.empty_bias,
                self.data.result_1,
                [],
                [],
            ),
        )

    def test_matmul_impl_bfloat16_postop(self):
        self.data.create_unittest_data("float32")
        with self.assertRaises(RuntimeError) as context:
            bias_as_postop = self.data.x.clone().to(torch.bfloat16)
            post_op_add = 6
            zentorch._C.zentorch_matmul_impl(
                self.data.x,
                self.data.x,
                self.data.empty_bias,
                self.data.result.to(self.data.x.dtype),
                [post_op_add],
                [bias_as_postop],
            )

        self.assertTrue(
            "zentorch_matmul only supports Float post ops "
            "when input matrix is Float" in str(context.exception)
        )

    def test_matmul_impl_int_postop(self):
        self.skip_if_bfloat16_unsupported_hardware()
        self.data.create_unittest_data("bfloat16")
        with self.assertRaises(RuntimeError) as context_int:
            bias_as_postop = self.data.x.clone().to(torch.int)
            post_op_add = 6
            zentorch._C.zentorch_matmul_impl(
                self.data.x,
                self.data.x,
                self.data.empty_bias,
                self.data.result.to(self.data.x.dtype),
                [post_op_add],
                [bias_as_postop],
            )

        self.assertTrue(
            "zentorch_matmul only supports BFloat16 post ops "
            "when input matrix is BFloat16" in str(context_int.exception)
        )

    def test_int_matmul_impl_postop(self):
        self.data.create_unittest_data("int")
        with self.assertRaises(RuntimeError) as context_int:
            bias_as_postop = self.data.x3d.clone().to(torch.int)
            post_op_add = 6
            zentorch._C.zentorch_matmul_impl(
                self.data.x,
                self.data.x,
                self.data.empty_bias,
                self.data.result.to(self.data.x.dtype),
                [post_op_add],
                [bias_as_postop],
            )

        self.assertTrue(
            "zentorch_matmul only supports Float and BFloat16"
            in str(context_int.exception)
        )


if __name__ == "__main__":
    run_tests()
