# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
from itertools import product
import torch
from parameterized import parameterized
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    zentorch,
    run_tests,
    qlinear_dtypes,
    input_dim_opt,
    q_weight_list_opt,
    bias_opt,
    q_granularity_opt,
    q_zero_points_dtype_opt,
    q_linear_dtype_opt,
)
from quant_utils import qdq_linear  # noqa: 402


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Qlinear_Binary(Zentorch_TestCase):
    @torch.inference_mode()
    def test_qlinear_unsupported_hardware(self):
        if zentorch._C.is_avx512_supported():
            self.skipTest(
                "Skipping hardware test, as AVX512 instructions are supported."
            )
        self.data.create_unittest_data("float32")

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear_mul_add(
                self.data.x_for_qlinear["float32"][2],
                self.data.y_int8[0],
                self.data.bias_for_qlinear[0],
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"]["float32"]["uint8"],
                self.data.y_scales["per_channel"],
                self.data.y_zero_points["per_channel"],
                self.data.binary_input[2],
                self.data.binary_input[2],
                self.data.x_for_qlinear["float32"][2].dtype,
            )
        self.assertTrue(
            "Zentorch's INT8 kernels require the CPU to support AVX512 instructions."
            in str(context.exception)
        )

    @parameterized.expand(
        product(
            qlinear_dtypes,
            input_dim_opt,
            q_weight_list_opt,
            bias_opt,
            q_granularity_opt,
            q_zero_points_dtype_opt,
            q_linear_dtype_opt,
        ),
        skip_on_empty=True,
    )
    @torch.inference_mode()
    def test_qlinear_mul_add_op(
        self,
        dtype,
        input_dim,
        q_weight_idx,
        bias_opt_idx,
        q_granularity_val,
        q_zero_points_dtype,
        input_dtype,
    ):
        self.skip_if_bfloat16_not_yet_supported(dtype)
        self.data.create_unittest_data(dtype)
        qdq_linear_mul_add_output = torch.add(
            qdq_linear(
                self.data.x_for_qlinear[input_dtype][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype],
                self.data.y_scales[q_granularity_val],
                self.data.y_zero_points[q_granularity_val],
            )
            * self.data.binary_input[input_dim],
            self.data.binary_input[input_dim],
        )
        # zentorch qlinear + eltwise fused op
        zentorch_qlinear_mul_add_output = torch.ops.zentorch.zentorch_qlinear_mul_add(
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype],
            self.data.y_scales[q_granularity_val],
            self.data.y_zero_points[q_granularity_val],
            self.data.binary_input[input_dim],
            self.data.binary_input[input_dim],
            torch.float32,
        )
        self.assertEqual(
            qdq_linear_mul_add_output,
            zentorch_qlinear_mul_add_output,
            atol=1e-3,
            rtol=1e-3,
        )


if __name__ == "__main__":
    run_tests()
