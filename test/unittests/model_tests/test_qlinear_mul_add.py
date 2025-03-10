# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import copy
import unittest
from itertools import product
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
    run_tests,
    reset_dynamo,
    qlinear_dtypes,
    input_dim_opt,
    q_weight_list_opt,
    bias_opt,
    q_granularity_opt,
    q_zero_points_dtype_opt,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Qlinear_Mul_Add(nn.Module):
    def __init__(self) -> None:
        super(Custom_Model_Qlinear_Mul_Add, self).__init__()

    def forward(
        self,
        inp,
        weight,
        bias,
        inp_scales,
        inp_zero_points,
        weight_scales,
        weight_zero_points,
        mul_input,
        add_input,
    ):
        qlinear_output = torch.ops.zentorch.zentorch_qlinear(
            inp,
            weight,
            bias,
            inp_scales,
            inp_zero_points,
            weight_scales,
            weight_zero_points,
            inp.dtype,
        )
        mul_output = torch.mul(qlinear_output, mul_input)
        add_output = torch.add(mul_output, add_input)
        return add_output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Qlinear_Mul_Add_Model(Zentorch_TestCase):
    @parameterized.expand(
        product(
            qlinear_dtypes,
            input_dim_opt,
            q_weight_list_opt,
            bias_opt,
            q_granularity_opt,
            q_zero_points_dtype_opt,
        ),
        skip_on_empty=True,
    )
    @torch.inference_mode()
    def test_qlinear_eltwise_model(
        self,
        dtype,
        input_dim,
        q_weight_idx,
        bias_opt_idx,
        q_granularity_val,
        q_zero_points_dtype,
    ):
        self.skip_if_bfloat16_not_yet_supported(dtype)
        self.data.create_unittest_data(dtype)

        model = Custom_Model_Qlinear_Mul_Add()
        zentorch_model = copy.deepcopy(model)

        model_output = model(
            self.data.x_for_qlinear["float32"][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
            self.data.y_scales[q_granularity_val],
            self.data.y_zero_points[q_granularity_val],
            self.data.binary_input[input_dim],
            self.data.binary_input[input_dim],
        )

        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_qlinear_mul_add"], 0)

        reset_dynamo()
        zentorch_model = torch.compile(zentorch_model, backend="zentorch")

        zentorch_output = zentorch_model(
            self.data.x_for_qlinear["float32"][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
            self.data.y_scales[q_granularity_val],
            self.data.y_zero_points[q_granularity_val],
            self.data.binary_input[input_dim],
            self.data.binary_input[input_dim],
        )
        self.assertEqual(counters["zentorch"]["pattern_matcher_qlinear_mul_add"], 1)
        self.assertEqual(model_output, zentorch_output)


if __name__ == "__main__":
    run_tests()
