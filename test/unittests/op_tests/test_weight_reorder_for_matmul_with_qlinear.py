# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import copy
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
    bias_opt,
    q_granularity_opt,
    q_zero_points_dtype_opt,
    q_linear_dtype_opt,
)
from quant_utils import qdq_linear  # noqa: 402


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Weight_Reorder_For_Matmul_With_Qlinear(Zentorch_TestCase):
    @parameterized.expand(
        product(
            qlinear_dtypes,
            input_dim_opt,
            bias_opt,
            q_granularity_opt,
            q_zero_points_dtype_opt,
            q_linear_dtype_opt,
            q_linear_dtype_opt,
        ),
        skip_on_empty=True,
    )
    @torch.inference_mode()
    def test_weight_reorder_for_matmul_with_qlinear_for_OCxIC(
        self,
        dtype,
        input_dim,
        bias_opt_idx,
        q_granularity_val,
        q_zero_points_dtype,
        input_dtype,
        output_dtype,
    ):
        self.data.create_unittest_data(dtype)
        self.skip_if_does_not_support_arg_combination_for_qlinear(
            bias_opt_idx, input_dtype, output_dtype
        )
        if (
            self.data.x_zero_points["per_tensor"][input_dtype][
                q_zero_points_dtype
            ].dtype
            == torch.int8
        ):
            self.skipTest(
                "Skipping test, if input zero points dtype is int8 as weight reorder "
                "is not supported with int8 inputs/activations."
            )

        # simulated qlinear
        qdq_linear_output = qdq_linear(
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[1],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype],
            self.data.y_scales[q_granularity_val],
            self.data.y_zero_points[q_granularity_val],
            None,
            self.data.get_torch_type(output_dtype),
            self.data.output_scales["per_tensor"][output_dtype]["positive_scales"],
            self.data.output_zero_points["per_tensor"][output_dtype],
        )

        weight = copy.deepcopy(self.data.y_int8[1])
        weight = torch.ops.zentorch.zentorch_weight_reorder_for_matmul.default(
            weight, is_weight_oc_x_ic=True
        )
        # zentorch qlinear
        zentorch_qlinear_output = torch.ops.zentorch.zentorch_qlinear(
            self.data.x_for_qlinear[input_dtype][input_dim],
            weight,
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
            qdq_linear_output, zentorch_qlinear_output, atol=1e-2, rtol=1e-2
        )

    @parameterized.expand(
        product(
            qlinear_dtypes,
            input_dim_opt,
            bias_opt,
            q_granularity_opt,
            q_zero_points_dtype_opt,
            q_linear_dtype_opt,
            q_linear_dtype_opt,
        ),
        skip_on_empty=True,
    )
    @torch.inference_mode()
    def test_weight_reorder_for_matmul_with_qlinear_for_ICxOC(
        self,
        dtype,
        input_dim,
        bias_opt_idx,
        q_granularity_val,
        q_zero_points_dtype,
        input_dtype,
        output_dtype,
    ):
        self.data.create_unittest_data(dtype)
        self.skip_if_does_not_support_arg_combination_for_qlinear(
            bias_opt_idx, input_dtype, output_dtype
        )
        if (
            self.data.x_zero_points["per_tensor"][input_dtype][
                q_zero_points_dtype
            ].dtype
            == torch.int8
        ):
            self.skipTest(
                "Skipping test, if input zero points dtype is int8 as weight reorder "
                "is not supported with int8 inputs/activations."
            )

        # simulated qlinear
        qdq_linear_output = qdq_linear(
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[1],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype],
            self.data.y_scales[q_granularity_val],
            self.data.y_zero_points[q_granularity_val],
            None,
            self.data.get_torch_type(output_dtype),
            self.data.output_scales["per_tensor"][output_dtype]["positive_scales"],
            self.data.output_zero_points["per_tensor"][output_dtype],
        )

        weight = copy.deepcopy(self.data.y_int8[1].t().contiguous())
        weight = torch.ops.zentorch.zentorch_weight_reorder_for_matmul.default(
            weight, is_weight_oc_x_ic=False
        )
        # zentorch qlinear
        zentorch_qlinear_output = torch.ops.zentorch.zentorch_qlinear(
            self.data.x_for_qlinear[input_dtype][input_dim],
            weight.t(),
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
            qdq_linear_output, zentorch_qlinear_output, atol=1e-2, rtol=1e-2
        )


if __name__ == "__main__":
    run_tests()
