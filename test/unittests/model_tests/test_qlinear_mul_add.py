# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import copy
import unittest
import torch
from torch import nn
import sys
from pathlib import Path
import zentorch

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    QLinearTestCase,
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
    q_linear_dtype_opt,
    get_comp_zero_points,
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
        output_dtype,
    ):
        qlinear_output = torch.ops.zentorch.zentorch_qlinear(
            inp,
            weight,
            bias,
            inp_scales,
            inp_zero_points,
            weight_scales,
            weight_zero_points,
            output_dtype=output_dtype,
        )
        mul_output = torch.mul(qlinear_output, mul_input)
        add_output = torch.add(mul_output, add_input)
        return add_output.to(output_dtype)


# stacked decorators get excecuted down->up
@unittest.skipIf(not zentorch._C.is_avx512_supported(), "No bf16 support on hardware")
@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
# temporary fix for Milan unittest failure
# TODO: Re-factor the unit-test configuration to use dtypes
# based on the underlying machine capabilities.
class Test_Qlinear_Mul_Add_Model(QLinearTestCase):
    @QLinearTestCase.hypothesis_params_qlinear_itr(
        dtype_list=["float32", "bfloat16"],  # adds missing combinations (not None bias)
        input_dim_opt_list=input_dim_opt,
        q_weight_list_opt_list=q_weight_list_opt,
        bias_opt_list=bias_opt,
        q_granularity_opt_list=q_granularity_opt,
        q_zero_points_dtype_opt_list=q_zero_points_dtype_opt,
        q_linear_dtype_opt_list=q_linear_dtype_opt,
        q_linear_output_dtype_opt_list=["float32", "bfloat16"]
    )
    @torch.inference_mode()
    def test_qlinear_mul_add_model(
        self,
        dtype,
        input_dim,
        q_weight_idx,
        bias_opt_idx,
        q_granularity_val,
        q_zero_points_dtype,
        input_dtype,
        output_dtype,
    ):
        self.skip_if_does_not_support_arg_combination_for_qlinear(
            bias_opt_idx, input_dtype, output_dtype
        )

        if (
            self.data.bias_for_qlinear[bias_opt_idx] is not None
            and self.data.bias_for_qlinear[bias_opt_idx].dtype
            != self.data.binary_input[input_dim].dtype
        ):
            self.skipTest("Skipping test, bias dtype has to match post-ops dtype.")

        model = Custom_Model_Qlinear_Mul_Add()
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
            self.data.binary_input[input_dim],
            self.data.binary_input[input_dim],
            output_dtype=self.data.get_torch_type(output_dtype),
        )

        counters.clear()
        self.assertEqual(counters["zentorch"]["pattern_matcher_qlinear_mul_add"], 0)

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
            self.data.binary_input[input_dim],
            self.data.binary_input[input_dim],
            output_dtype=self.data.get_torch_type(output_dtype),
        )
        self.assertEqual(counters["zentorch"]["pattern_matcher_qlinear_mul_add"], 1)
        self.assertEqual(model_output, zentorch_output, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
