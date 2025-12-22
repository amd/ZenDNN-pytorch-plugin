# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    QLinearTestCase,
    has_zentorch,
    run_tests,
    qlinear_dtypes,
    input_dim_opt,
    q_weight_list_opt,
    bias_opt,
    q_granularity_opt,
    q_zero_points_dtype_opt,
    q_linear_dtype_opt,
    get_comp_zero_points,
)
from quant_utils import qdq_linear  # noqa: 402


# TODO:Segregate DLRM and LLM specific, model tests and/or test data.
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
        output_dtype,
        output_scales=None,
        output_zero_points=None,
        use_zentorch=False,
        qlinear_output=None,
    ):
        if use_zentorch:
            # zentorch qlinear
            if qlinear_output is None:
                qlinear_output = torch.ops.zentorch.zentorch_qlinear(
                    inp,
                    weight,
                    bias,
                    inp_scales,
                    inp_zero_points,
                    weight_scales,
                    weight_zero_points,
                    output_dtype=output_dtype,
                    output_scales=output_scales,
                    output_zero_points=output_zero_points,
                )
            else:
                torch.ops.zentorch.zentorch_qlinear.out(
                    qlinear_output,
                    inp,
                    weight,
                    bias,
                    inp_scales,
                    inp_zero_points,
                    weight_scales,
                    weight_zero_points,
                    output_dtype=output_dtype,
                    output_scales=output_scales,
                    output_zero_points=output_zero_points,
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
                None,
                output_dtype,
                output_scales,
                output_zero_points,
            )

        return qlinear_output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Qlinear_Model(QLinearTestCase):
    @QLinearTestCase.hypothesis_params_qlinear_itr(
        dtype_list=qlinear_dtypes,
        input_dim_opt_list=input_dim_opt,
        q_weight_list_opt_list=q_weight_list_opt,
        bias_opt_list=bias_opt,
        q_granularity_opt_list=q_granularity_opt,
        q_zero_points_dtype_opt_list=q_zero_points_dtype_opt,
        q_linear_dtype_opt_list=q_linear_dtype_opt,
        q_linear_output_dtype_opt_list=q_linear_dtype_opt
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
        input_dtype,
        output_dtype,
    ):
        # TODO
        # Enable input type to be bfloat16 once we have support for bf16 to int8 quantization
        # ZENAI-1322
        if input_dtype == "bfloat16":
            self.skipTest(
                "Skipping test, if input dtype is bfloat16, then it is not supported."
            )

        # adding all the skip conditions (there are 4)
        self.skip_if_does_not_support_arg_combination_for_qlinear(
            bias_opt_idx, input_dtype, output_dtype
        )

        model = Custom_Model_Qlinear()

        simulated_output = model(
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype],
            self.data.y_scales[q_granularity_val],
            self.data.y_zero_points[q_granularity_val],
            self.data.get_torch_type(output_dtype),
            self.data.output_scales["per_tensor"][output_dtype]["positive_scales"],
            self.data.output_zero_points["per_tensor"][output_dtype],
        )

        zentorch_output = model(
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            get_comp_zero_points(
                self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype]
            ),
            self.data.y_scales[q_granularity_val],
            get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
            self.data.get_torch_type(output_dtype),
            self.data.output_scales["per_tensor"][output_dtype]["positive_scales"],
            get_comp_zero_points(self.data.output_zero_points["per_tensor"][output_dtype]),
            use_zentorch=True,
        )
        self.assertEqual(simulated_output, zentorch_output, atol=1e-2, rtol=1e-2)

        output_torch_dtype = self.data.get_torch_type(output_dtype)
        input = self.data.binary_input[input_dim].to(output_torch_dtype)
        simulated_cat_output = torch.cat([input, simulated_output], dim=(input_dim - 1))
        zentorch_cat_output = torch.ones_like(simulated_cat_output).to(
            output_torch_dtype
        )
        cat_output_stride = zentorch_cat_output.stride()
        input_view = torch.as_strided(
            zentorch_cat_output, size=input.shape, stride=cat_output_stride
        )
        input_view.copy_(input)

        qlinear_view = torch.as_strided(
            zentorch_cat_output,
            size=simulated_output.shape,
            stride=cat_output_stride,
            storage_offset=input.shape[-1],
        )

        model(
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            get_comp_zero_points(
                self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype]
            ),
            self.data.y_scales[q_granularity_val],
            get_comp_zero_points(self.data.y_zero_points[q_granularity_val].to(torch.int32)),
            output_torch_dtype,
            self.data.output_scales["per_tensor"][output_dtype]["positive_scales"],
            get_comp_zero_points(self.data.output_zero_points["per_tensor"][output_dtype]),
            use_zentorch=True,
            qlinear_output=qlinear_view,
        )

        self.assertEqual(
            simulated_cat_output, zentorch_cat_output, atol=1e-2, rtol=1e-2
        )


if __name__ == "__main__":
    run_tests()
