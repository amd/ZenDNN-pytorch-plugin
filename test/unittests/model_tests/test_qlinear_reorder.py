# ******************************************************************************
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import copy
import unittest
import torch
from torch.testing import FileCheck
from torch import nn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    QLinearTestCase,
    counters,
    has_zentorch,
    reset_dynamo,
    run_tests,
    input_dim_opt,
    qlinear_dtypes,
    bias_opt,
    q_granularity_opt,
    q_zero_points_dtype_opt,
    get_comp_zero_points,
    freeze_opt,
    cpp_wrapper_opt,
    test_with_freeze_opt_and_cpp_wrapper,
)


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Custom_Model_Zentorch_Qlinear_X3(nn.Module):
    def __init__(self) -> None:
        super(Custom_Model_Zentorch_Qlinear_X3, self).__init__()

    def forward(
        self,
        input,
        weights,
        input_scales,
        input_zero_points,
        weight_scales,
        weight_zero_points,
        biases,
        output_dtype,
    ):
        qlinear_0 = torch.ops.zentorch.zentorch_qlinear(
            input,
            weights,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            biases,
            None,
            None,
            output_dtype,
        )
        qlinear_1 = torch.ops.zentorch.zentorch_qlinear(
            qlinear_0,
            weights,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            biases,
            None,
            None,
            output_dtype,
        )
        qlinear_2 = torch.ops.zentorch.zentorch_qlinear(
            qlinear_1,
            weights,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            biases,
            None,
            None,
            output_dtype,
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
        input_scales,
        input_zero_points,
        weight_scales,
        weight_zero_points,
        biases,
        output_dtype,
    ):
        qlinear_0 = torch.ops.zentorch.zentorch_qlinear(
            input,
            weights,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            biases,
            None,
            None,
            output_dtype,
        )
        qlinear_1 = torch.ops.zentorch.zentorch_qlinear_relu(
            qlinear_0,
            weights,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            biases,
            None,
            None,
            output_dtype,
        )
        qlinear_2 = torch.ops.zentorch.zentorch_qlinear_sigmoid(
            qlinear_1,
            weights,
            input_scales,
            input_zero_points,
            weight_scales,
            weight_zero_points,
            biases,
            None,
            None,
            output_dtype,
        )
        return qlinear_2


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Qlinear_Model(QLinearTestCase):
    @QLinearTestCase.hypothesis_params_qlinear_itr(
        dtype_list=qlinear_dtypes,
        input_dim_opt_list=input_dim_opt,
        bias_opt_list=bias_opt,
        q_granularity_opt_list=q_granularity_opt,
        q_zero_points_dtype_opt_list=q_zero_points_dtype_opt,
        freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
        # cold cpp_wrapper compile exceeds the default deadline; see
        # test_with_freeze_opt_and_cpp_wrapper in zentorch_test_utils.
        time_out=60000,
    )
    @torch.inference_mode()
    def test_qlinear_mix_x3(
        self,
        dtype,
        input_dim,
        bias_opt_idx,
        q_granularity_val,
        q_zero_points_dtype,
        freeze_opt,
        cpp_wrapper,
    ):

        model = Custom_Model_Zentorch_Qlinear_Mix_X3()
        zentorch_model = copy.deepcopy(model)

        model_output = model(
            self.data.x_for_qlinear[dtype][input_dim],
            self.data.y_int8_square[0],
            self.data.x_scales["per_tensor"],
            get_comp_zero_points(
                self.data.x_zero_points["per_tensor"][dtype][q_zero_points_dtype]
            ),
            self.data.y_scales_square[q_granularity_val],
            get_comp_zero_points(self.data.y_zero_points_square[q_granularity_val]),
            self.data.bias_for_qlinear_square[bias_opt_idx],
            self.data.get_torch_type(dtype),
        )

        counters.clear()
        self.assertEqual(counters["zentorch"]["optimized_reorder"], 0)

        reset_dynamo()
        zentorch_model = torch.compile(zentorch_model, backend="zentorch")
        zentorch_output, cpp_code = test_with_freeze_opt_and_cpp_wrapper(
            zentorch_model,
            (
                self.data.x_for_qlinear[dtype][input_dim],
                self.data.y_int8_square[0],
                self.data.x_scales["per_tensor"],
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"][dtype][q_zero_points_dtype]
                ),
                self.data.y_scales_square[q_granularity_val],
                get_comp_zero_points(self.data.y_zero_points_square[q_granularity_val]),
                self.data.bias_for_qlinear_square[bias_opt_idx],
                self.data.get_torch_type(dtype),
            ),
            freeze_opt,
            cpp_wrapper,
        )
        self.assertEqual(counters["zentorch"]["optimized_reorder"], 2)
        # Prepack is gated on missing weight zp (not input zp). These models
        # always pass int8 weight zp, which get_comp_zero_points maps to None.
        # Mix graph: only the first op is zentorch_qlinear (relu/sigmoid are not
        # prepacked).
        prepacked_weights = 1 if freeze_opt else 0
        self.assertEqual(
            counters["zentorch"]["zentorch_weight_prepack_for_dynamic_qlinear"],
            prepacked_weights,
        )
        self.assertEqual(model_output, zentorch_output, atol=1e-2, rtol=1e-2)
        # Pillar 2 (codegen): op lowers to its AOTI C-shim (see helper docstring).
        if cpp_wrapper:
            FileCheck().check("aoti_torch_cpu_zentorch").run(cpp_code)

    @QLinearTestCase.hypothesis_params_qlinear_itr(
        dtype_list=qlinear_dtypes,
        input_dim_opt_list=input_dim_opt,
        bias_opt_list=bias_opt,
        q_granularity_opt_list=q_granularity_opt,
        q_zero_points_dtype_opt_list=q_zero_points_dtype_opt,
        freeze_list=freeze_opt,
        cpp_wrapper_opt_list=cpp_wrapper_opt,
        # cold cpp_wrapper compile exceeds the default deadline; see
        # test_with_freeze_opt_and_cpp_wrapper in zentorch_test_utils.
        time_out=60000,
    )
    @torch.inference_mode()
    def test_qlinear_x3(
        self,
        dtype,
        input_dim,
        bias_opt_idx,
        q_granularity_val,
        q_zero_points_dtype,
        freeze_opt,
        cpp_wrapper,
    ):

        model = Custom_Model_Zentorch_Qlinear_X3()
        zentorch_model = copy.deepcopy(model)

        model_output = model(
            self.data.x_for_qlinear[dtype][input_dim],
            self.data.y_int8_square[0],
            self.data.x_scales["per_tensor"],
            get_comp_zero_points(
                self.data.x_zero_points["per_tensor"][dtype][q_zero_points_dtype]
            ),
            self.data.y_scales_square[q_granularity_val],
            get_comp_zero_points(self.data.y_zero_points_square[q_granularity_val]),
            self.data.bias_for_qlinear_square[bias_opt_idx],
            self.data.get_torch_type(dtype),
        )

        counters.clear()
        self.assertEqual(counters["zentorch"]["optimized_reorder"], 0)

        reset_dynamo()
        zentorch_model = torch.compile(zentorch_model, backend="zentorch")

        zentorch_output, cpp_code = test_with_freeze_opt_and_cpp_wrapper(
            zentorch_model,
            (
                self.data.x_for_qlinear[dtype][input_dim],
                self.data.y_int8_square[0],
                self.data.x_scales["per_tensor"],
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"][dtype][q_zero_points_dtype]
                ),
                self.data.y_scales_square[q_granularity_val],
                get_comp_zero_points(self.data.y_zero_points_square[q_granularity_val]),
                self.data.bias_for_qlinear_square[bias_opt_idx],
                self.data.get_torch_type(dtype),
            ),
            freeze_opt,
            cpp_wrapper,
        )
        self.assertEqual(counters["zentorch"]["optimized_reorder"], 2)
        # Prepack is gated on missing weight zp (not input zp). These models
        # always pass int8 weight zp, which get_comp_zero_points maps to None.
        # Three zentorch_qlinear ops → three prepacks when frozen.
        prepacked_weights = 3 if freeze_opt else 0
        self.assertEqual(
            counters["zentorch"]["zentorch_weight_prepack_for_dynamic_qlinear"],
            prepacked_weights,
        )
        self.assertEqual(model_output, zentorch_output, atol=1e-2, rtol=1e-2)
        # Pillar 2 (codegen): op lowers to its AOTI C-shim (see helper docstring).
        if cpp_wrapper:
            FileCheck().check("aoti_torch_cpu_zentorch").run(cpp_code)


if __name__ == "__main__":
    run_tests()
