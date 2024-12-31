# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

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
    has_zentorch,
    run_tests,
    supported_dtypes,
    input_dim_opt,
    q_weight_list_opt,
    bias_opt,
    q_granularity_opt,
    q_zero_points_dtype_opt,
)
from quant_utils import qdq_linear  # noqa: 402


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Qlinear(nn.Module):
    def __init__(self) -> None:
        super(Custom_Model_Qlinear, self).__init__()

    def forward(
        self,
        inp,
        weight,
        bias,
        inp_scales,
        inp_zero_points,
        weight_scales,
        weight_zero_points,
        use_zentorch=False,
    ):
        if use_zentorch:
            # zentorch qlinear
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
        else:
            # simulated qlinear
            qlinear_output = qdq_linear(
                inp,
                weight,
                bias,
                inp_scales,
                inp_zero_points,
                weight_scales,
                weight_zero_points,
            )

        return qlinear_output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Qlinear_Model(Zentorch_TestCase):
    @parameterized.expand(
        product(
            supported_dtypes,
            input_dim_opt,
            q_weight_list_opt,
            bias_opt,
            q_granularity_opt,
            q_zero_points_dtype_opt,
        )
    )
    @torch.inference_mode()
    def test_qlinear_model(
        self,
        dtype,
        input_dim,
        q_weight_idx,
        bias_opt_idx,
        q_granularity_val,
        q_zero_points_dtype,
    ):
        self.skip_if_bfloat16_not_yet_supported(dtype)
        self.data.create_data(dtype)

        model = Custom_Model_Qlinear()

        simulated_output = model(
            self.data.x_for_qlinear[input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][q_zero_points_dtype],
            self.data.y_scales[q_granularity_val],
            self.data.y_zero_points[q_granularity_val],
        )
        zentorch_output = model(
            self.data.x_for_qlinear[input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][q_zero_points_dtype],
            self.data.y_scales[q_granularity_val],
            self.data.y_zero_points[q_granularity_val],
            use_zentorch=True,
        )
        self.assertEqual(simulated_output, zentorch_output, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    run_tests()
