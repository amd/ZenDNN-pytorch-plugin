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
    run_tests,
    qlinear_dtypes,
    input_dim_opt,
    q_weight_list_opt,
    bias_opt,
    q_granularity_opt,
    q_zero_points_dtype_opt,
    q_linear_dtype_opt,
    qlinear_eltwise_map,
)
from quant_utils import qdq_linear  # noqa: 402


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Qlinear_Eltwise(Zentorch_TestCase):
    @parameterized.expand(
        product(
            qlinear_dtypes,
            input_dim_opt,
            q_weight_list_opt,
            bias_opt,
            q_granularity_opt,
            q_zero_points_dtype_opt,
            q_linear_dtype_opt,
            q_linear_dtype_opt,
            qlinear_eltwise_map.keys(),
        ),
        skip_on_empty=True,
    )
    @torch.inference_mode()
    def test_qlinear_eltwise_fused_op_accuracy(
        self,
        dtype,
        input_dim,
        q_weight_idx,
        bias_opt_idx,
        q_granularity_val,
        q_zero_points_dtype,
        input_dtype,
        output_dtype,
        eltwise_op,
    ):
        self.data.create_unittest_data(dtype)
        self.skip_if_does_not_support_arg_combination_for_qlinear(
            bias_opt_idx, input_dtype, output_dtype
        )

        # simulated qlinear + eltwise op
        qdq_linear_eltwise_output = qdq_linear(
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype],
            self.data.y_scales[q_granularity_val],
            self.data.y_zero_points[q_granularity_val],
            qlinear_eltwise_map[eltwise_op][0].eval(),
            self.data.get_torch_type(output_dtype),
            self.data.output_scales["per_tensor"][output_dtype]["positive_scales"],
            self.data.output_zero_points["per_tensor"][output_dtype],
        )

        # zentorch qlinear + eltwise fused op
        zentorch_qlinear_eltwise_output = qlinear_eltwise_map[eltwise_op][1](
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype],
            self.data.y_scales[q_granularity_val],
            self.data.y_zero_points[q_granularity_val],
            output_dtype=self.data.get_torch_type(output_dtype),
            output_scales=self.data.output_scales["per_tensor"][output_dtype][
                "positive_scales"
            ],
            output_zero_points=self.data.output_zero_points["per_tensor"][output_dtype],
        )

        self.assertEqual(
            qdq_linear_eltwise_output,
            zentorch_qlinear_eltwise_output,
            atol=1e-2,
            rtol=1e-2,
        )


if __name__ == "__main__":
    run_tests()
