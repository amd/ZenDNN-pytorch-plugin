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
    qlinear_eltwise_map,
    q_linear_dtype_opt,
    get_comp_zero_points,
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
        output_torch_dtype,
    ):
        qlinear_output = torch.ops.zentorch.zentorch_qlinear(
            inp,
            weight,
            bias,
            inp_scales,
            inp_zero_points,
            weight_scales,
            weight_zero_points,
            output_dtype=output_torch_dtype,
        )
        eltwise_output = self.eltwise_op(qlinear_output)

        return eltwise_output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Qlinear_Eltwise_Model(Zentorch_TestCase):
    @parameterized.expand(
        product(
            qlinear_dtypes,
            q_linear_dtype_opt,
            input_dim_opt,
            q_weight_list_opt,
            bias_opt,
            q_granularity_opt,
            q_zero_points_dtype_opt,
            qlinear_eltwise_map.keys(),
            ["float32", "bfloat16"],  # o/p dtype is float only in this case
        ),
        skip_on_empty=True,
    )
    @torch.inference_mode()
    def test_qlinear_eltwise_model(
        self,
        dtype,
        input_dtype,
        input_dim,
        q_weight_idx,
        bias_opt_idx,
        q_granularity_val,
        q_zero_points_dtype,
        eltwise_op,
        output_dtype,
    ):
        self.data.create_unittest_data(dtype)
        if (
            self.data.bias_for_qlinear[bias_opt_idx] is not None
            and input_dtype in ("float32", "bfloat16")
            and self.data.bias_for_qlinear[bias_opt_idx].dtype
            != self.data.get_torch_type(input_dtype)
        ):
            self.skipTest(
                "Skipping test, if bias is not None and input is floating-point, then "
                "bias dtype has to match input dtype"
            )

        model = Custom_Model_Qlinear_Eltwise(qlinear_eltwise_map[eltwise_op][0])
        zentorch_model = copy.deepcopy(model)
        model_output = model(
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            get_comp_zero_points(
                self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype]
            ),
            self.data.y_scales[q_granularity_val],
            get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
            output_torch_dtype=self.data.get_torch_type(output_dtype),
        )

        counters.clear()
        self.assertEqual(counters["zentorch"][eltwise_op + "_fusion"], 0)

        reset_dynamo()
        zentorch_model = torch.compile(zentorch_model, backend="zentorch")

        zentorch_output = zentorch_model(
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            get_comp_zero_points(
                self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype]
            ),
            self.data.y_scales[q_granularity_val],
            get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
            output_torch_dtype=self.data.get_torch_type(output_dtype),
        )
        self.assertEqual(counters["zentorch"][eltwise_op + "_fusion"], 1)
        self.assertEqual(model_output, zentorch_output)


if __name__ == "__main__":
    run_tests()
