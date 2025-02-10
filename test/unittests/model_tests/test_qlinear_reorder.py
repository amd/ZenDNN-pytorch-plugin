# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import copy
from itertools import product
import unittest
import torch
from parameterized import parameterized
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    counters,
    has_zentorch,
    reset_dynamo,
    run_tests,
    input_dim_opt,
    qlinear_dtypes,
    bias_opt,
    q_granularity_opt,
    q_zero_points_dtype_opt,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Zentorch_Qlinear_X3(nn.Module):
    def __init__(self) -> None:
        super(Custom_Model_Zentorch_Qlinear_X3, self).__init__()

    def forward(
        self,
        input,
        weights,
        biases,
        input_scales,
        input_zero_points,
        weight_scales,
        weight_zero_points,
        output_dtype,
    ):
        qlinear_0 = torch.ops.zentorch.zentorch_qlinear(
            input,
            weights,
            biases,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            output_dtype=output_dtype,
        )
        qlinear_1 = torch.ops.zentorch.zentorch_qlinear(
            qlinear_0,
            weights,
            biases,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            output_dtype=output_dtype,
        )
        qlinear_2 = torch.ops.zentorch.zentorch_qlinear(
            qlinear_1,
            weights,
            biases,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            output_dtype=output_dtype,
        )
        return qlinear_2


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Zentorch_Qlinear_Mix_X3(nn.Module):
    def __init__(self) -> None:
        super(Custom_Model_Zentorch_Qlinear_Mix_X3, self).__init__()

    def forward(
        self,
        input,
        weights,
        biases,
        input_scales,
        input_zero_points,
        weight_scales,
        weight_zero_points,
        output_dtype,
    ):
        qlinear_0 = torch.ops.zentorch.zentorch_qlinear(
            input,
            weights,
            biases,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            output_dtype=output_dtype,
        )
        qlinear_1 = torch.ops.zentorch.zentorch_qlinear_relu(
            qlinear_0,
            weights,
            biases,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            output_dtype=output_dtype,
        )
        qlinear_2 = torch.ops.zentorch.zentorch_qlinear_sigmoid(
            qlinear_1,
            weights,
            biases,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            output_dtype=output_dtype,
        )
        return qlinear_2


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Qlinear_Model(Zentorch_TestCase):
    @parameterized.expand(
        product(
            qlinear_dtypes,
            input_dim_opt,
            bias_opt,
            q_granularity_opt,
            q_zero_points_dtype_opt,
        ),
        skip_on_empty=True,
    )
    @torch.inference_mode()
    def test_qlinear_x3(
        self,
        dtype,
        input_dim,
        bias_opt_idx,
        q_granularity_val,
        q_zero_points_dtype,
    ):
        self.data.create_unittest_data(dtype)

        model = Custom_Model_Zentorch_Qlinear_Mix_X3()
        zentorch_model = copy.deepcopy(model)

        model_output = model(
            self.data.x_for_qlinear[dtype][input_dim],
            self.data.y_int8_square[0],
            self.data.bias_for_qlinear_square[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][dtype][q_zero_points_dtype],
            self.data.y_scales_square[q_granularity_val],
            self.data.y_zero_points_square[q_granularity_val],
            self.data.get_torch_type(dtype),
        )

        counters.clear()
        self.assertEqual(counters["zentorch"]["optimized_reorder"], 0)

        reset_dynamo()
        zentorch_model = torch.compile(zentorch_model, backend="zentorch")

        zentorch_output = zentorch_model(
            self.data.x_for_qlinear[dtype][input_dim],
            self.data.y_int8_square[0],
            self.data.bias_for_qlinear_square[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][dtype][q_zero_points_dtype],
            self.data.y_scales_square[q_granularity_val],
            self.data.y_zero_points_square[q_granularity_val],
            self.data.get_torch_type(dtype),
        )
        self.assertEqual(counters["zentorch"]["optimized_reorder"], 2)
        self.assertEqual(model_output, zentorch_output)

    @parameterized.expand(
        product(
            qlinear_dtypes,
            input_dim_opt,
            bias_opt,
            q_granularity_opt,
            q_zero_points_dtype_opt,
        ),
        skip_on_empty=True,
    )
    @torch.inference_mode()
    def test_qlinear_mix_x3(
        self,
        dtype,
        input_dim,
        bias_opt_idx,
        q_granularity_val,
        q_zero_points_dtype,
    ):
        self.data.create_unittest_data(dtype)

        model = Custom_Model_Zentorch_Qlinear_X3()
        zentorch_model = copy.deepcopy(model)

        model_output = model(
            self.data.x_for_qlinear[dtype][input_dim],
            self.data.y_int8_square[0],
            self.data.bias_for_qlinear_square[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][dtype][q_zero_points_dtype],
            self.data.y_scales_square[q_granularity_val],
            self.data.y_zero_points_square[q_granularity_val],
            self.data.get_torch_type(dtype),
        )

        counters.clear()
        self.assertEqual(counters["zentorch"]["optimized_reorder"], 0)

        reset_dynamo()
        zentorch_model = torch.compile(zentorch_model, backend="zentorch")

        zentorch_output = zentorch_model(
            self.data.x_for_qlinear[dtype][input_dim],
            self.data.y_int8_square[0],
            self.data.bias_for_qlinear_square[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][dtype][q_zero_points_dtype],
            self.data.y_scales_square[q_granularity_val],
            self.data.y_zero_points_square[q_granularity_val],
            self.data.get_torch_type(dtype),
        )
        self.assertEqual(counters["zentorch"]["optimized_reorder"], 2)
        self.assertEqual(model_output, zentorch_output)


if __name__ == "__main__":
    run_tests()
