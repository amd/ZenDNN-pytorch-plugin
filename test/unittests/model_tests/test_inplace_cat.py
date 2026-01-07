# ******************************************************************************
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
from torch import nn
import sys
from pathlib import Path


sys.path.append(str(Path(__file__).parent.parent))
from quant_utils import qdq_linear  # noqa: E402
from unittest_utils import (  # noqa: 402
    QLinearTestCase,
    has_zentorch,
    run_tests,
    counters,
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
class Custom_Model_Linear_Cat(nn.Module):
    def __init__(self) -> None:
        super(Custom_Model_Linear_Cat, self).__init__()

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
    ):
        if use_zentorch:
            qlinear_output = torch.ops.zentorch.zentorch_qlinear(
                inp,
                weight,
                bias,
                inp_scales,
                get_comp_zero_points(inp_zero_points),
                weight_scales,
                get_comp_zero_points(weight_zero_points),
                output_dtype=output_dtype,
                output_scales=output_scales,
                output_zero_points=get_comp_zero_points(output_zero_points),
            )
            qlinear_relu_output = torch.ops.zentorch.zentorch_qlinear_relu(
                inp,
                weight,
                bias,
                inp_scales,
                get_comp_zero_points(inp_zero_points),
                weight_scales,
                get_comp_zero_points(weight_zero_points),
                output_dtype=output_dtype,
                output_scales=output_scales,
                output_zero_points=get_comp_zero_points(output_zero_points),
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
            qlinear_relu_output = qdq_linear(
                inp,
                weight,
                bias,
                inp_scales,
                inp_zero_points,
                weight_scales,
                weight_zero_points,
                torch.nn.ReLU(),
                output_dtype,
                output_scales,
                output_zero_points,
            )
        cat_output = torch.cat(
            [qlinear_relu_output, qlinear_output], dim=(inp.dim() - 1)
        )
        return cat_output


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Inplace_Cat_Model(QLinearTestCase):
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
    def test_inplace_cat_model(
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
        # adding all the skip conditions (there are 4)
        if (
            self.data.bias_for_qlinear[bias_opt_idx] is None
            and input_dtype in ("float32", "bfloat16")
            and output_dtype not in (input_dtype, "int8", "uint8")
        ):
            self.skipTest(
                "Skipping test, if bias is None and input is floating-point, then "
                "output dtype has to either match input dtype or be any of int8 "
                "or uint8"
            )

        if (
            self.data.bias_for_qlinear[bias_opt_idx] is not None
            and self.data.bias_for_qlinear[bias_opt_idx].dtype == torch.float32
            and output_dtype == "bfloat16"
        ):
            self.skipTest(
                "Skipping test, if bias is fp32, then output dtype cannot be bf16."
            )

        if (
            self.data.bias_for_qlinear[bias_opt_idx] is not None
            and self.data.bias_for_qlinear[bias_opt_idx].dtype == torch.bfloat16
            and output_dtype == "float32"
        ):
            self.skipTest(
                "Skipping test, if bias is bf16, then output dtype cannot be fp32."
            )

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

        model = Custom_Model_Linear_Cat()

        output_torch_dtype = self.data.get_torch_type(output_dtype)
        simulated_output = model(
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype],
            self.data.y_scales[q_granularity_val],
            self.data.y_zero_points[q_granularity_val],
            output_torch_dtype,
            self.data.output_scales["per_tensor"][output_dtype]["positive_scales"],
            self.data.output_zero_points["per_tensor"][output_dtype],
        )

        reset_dynamo()
        counters.clear()
        model = torch.compile(model, backend="zentorch")
        zentorch_output = model(
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype],
            self.data.y_scales[q_granularity_val],
            self.data.y_zero_points[q_granularity_val],
            output_torch_dtype,
            self.data.output_scales["per_tensor"][output_dtype]["positive_scales"],
            self.data.output_zero_points["per_tensor"][output_dtype],
            use_zentorch=True,
        )

        self.assertEqual(counters["zentorch"]["out_variant"], 2)
        self.assertEqual(simulated_output, zentorch_output, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    run_tests()
