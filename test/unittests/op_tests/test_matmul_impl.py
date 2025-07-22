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
    run_tests,
    supported_dtypes,
    zentorch,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Matmul_Impl_Op(MMTestCase):
    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=supported_dtypes
    )
    def test_matmul_impl_for_mv_and_dot(self, dtype):

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

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=["float32"]
    )
    def test_matmul_impl_bfloat16_postop(self, dtype):
        bias_as_postop = self.data.x.clone().to(torch.bfloat16)
        post_op_add = 6
        with self.assertRaises(RuntimeError) as context:
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

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=["bfloat16"]
    )
    def test_matmul_impl_int_postop(self, dtype):
        self.skip_if_bfloat16_unsupported_hardware()
        bias_as_postop = self.data.x.clone().to(torch.int)
        post_op_add = 6
        with self.assertRaises(RuntimeError) as context_int:
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

    @MMTestCase.hypothesis_params_mm_itr(
        dtype_list=["int"]
    )
    def test_int_matmul_impl_postop(self, dtype):
        bias_as_postop = self.data.x3d.clone().to(torch.int)
        post_op_add = 6
        with self.assertRaises(RuntimeError) as context_int:
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
