# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
import torch
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
    qlinear_eltwise_map,
    get_comp_zero_points,
)
from quant_utils import qdq_linear  # noqa: 402


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Qlinear_Eltwise(QLinearTestCase):
    @QLinearTestCase.hypothesis_params_qlinear_itr(
        dtype_list=qlinear_dtypes,
        input_dim_opt_list=input_dim_opt,
        q_weight_list_opt_list=q_weight_list_opt,
        bias_opt_list=bias_opt,
        q_granularity_opt_list=q_granularity_opt,
        q_zero_points_dtype_opt_list=q_zero_points_dtype_opt,
        q_linear_dtype_opt_list=q_linear_dtype_opt,
        q_linear_output_dtype_opt_list=q_linear_dtype_opt,
        qlinear_eltwise_opt_list=qlinear_eltwise_map.keys()
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
        # TODO
        # Enable input type to be bfloat16 once we have support for bf16 to int8 quantization
        # ZENAI-1322
        if input_dtype == "bfloat16":
            self.skipTest(
                "Skipping test, if input dtype is bfloat16, then it is not supported."
            )

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
            get_comp_zero_points(
                self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype]
            ),
            self.data.y_scales[q_granularity_val],
            get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
            output_dtype=self.data.get_torch_type(output_dtype),
            output_scales=self.data.output_scales["per_tensor"][output_dtype][
                "positive_scales"
            ],
            output_zero_points=get_comp_zero_points(
                self.data.output_zero_points["per_tensor"][output_dtype]
            ),
        )

        self.assertEqual(
            qdq_linear_eltwise_output,
            zentorch_qlinear_eltwise_output,
            atol=1e-2,
            rtol=1e-2,
        )


if __name__ == "__main__":
    run_tests()
