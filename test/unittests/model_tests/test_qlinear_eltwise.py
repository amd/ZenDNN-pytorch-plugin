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
    supported_dtypes,
    input_dim_opt,
    q_weight_list_opt,
    bias_opt,
    q_granularity_opt,
    q_zero_points_dtype_opt,
    qlinear_eltwise_map,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Qlinear_Eltwise(nn.Module):
    def __init__(self, eltwise_op) -> None:
        super(Custom_Model_Qlinear_Eltwise, self).__init__()
        self.eltwise_op = eltwise_op

    def forward(
        self,
        inp,
        weight,
        bias,
        inp_scales,
        inp_zero_points,
        weight_scales,
        weight_zero_points,
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
        eltwise_output = self.eltwise_op(qlinear_output)

        return eltwise_output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Qlinear_Eltwise_Model(Zentorch_TestCase):
    @parameterized.expand(
        product(
            supported_dtypes,
            input_dim_opt,
            q_weight_list_opt,
            bias_opt,
            q_granularity_opt,
            q_zero_points_dtype_opt,
            qlinear_eltwise_map.keys(),
        )
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
        eltwise_op,
    ):
        self.skip_if_bfloat16_not_yet_supported(dtype)
        self.data.create_data(dtype)

        model = Custom_Model_Qlinear_Eltwise(qlinear_eltwise_map[eltwise_op][0])
        zentorch_model = copy.deepcopy(model)

        model_output = model(
            self.data.x_for_qlinear[input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][q_zero_points_dtype],
            self.data.y_scales[q_granularity_val],
            self.data.y_zero_points[q_granularity_val],
        )

        counters.clear()
        self.assertEqual(counters["zentorch"][eltwise_op + "_fusion"], 0)

        reset_dynamo()
        zentorch_model = torch.compile(zentorch_model, backend="zentorch")

        zentorch_output = zentorch_model(
            self.data.x_for_qlinear[input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][q_zero_points_dtype],
            self.data.y_scales[q_granularity_val],
            self.data.y_zero_points[q_granularity_val],
        )
        self.assertEqual(counters["zentorch"][eltwise_op + "_fusion"], 1)
        self.assertEqual(model_output, zentorch_output)


if __name__ == "__main__":
    run_tests()
