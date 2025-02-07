# ******************************************************************************
# Copyright (c) 2025 Advanced Micro Devices, Inc.
# All rights reserved.
# ******************************************************************************

import unittest
from itertools import product
import torch
from parameterized import parameterized
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from unittest_utils import (  # noqa: 402
    Zentorch_TestCase,
    has_zentorch,
    zentorch,
    run_tests,
    qlinear_dtypes,
    input_dim_opt,
    q_weight_list_opt,
    bias_opt,
    q_granularity_opt,
    q_zero_points_dtype_opt,
    q_linear_dtype_opt,
)
from quant_utils import qdq_linear  # noqa: 402


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Qlinear(Zentorch_TestCase):
    @torch.inference_mode()
    def test_qlinear_unsupported_hardware(self):
        if zentorch._C.is_avx512_supported():
            self.skipTest(
                "Skipping hardware test, as AVX512 instructions are supported."
            )
        self.data.create_data("float32")

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][2],
                self.data.y_int8[0],
                self.data.bias_for_qlinear[0],
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"]["float32"]["uint8"],
                self.data.y_scales["per_channel"],
                self.data.y_zero_points["per_channel"],
                self.data.x_for_qlinear["float32"][2].dtype,
            )
        self.assertTrue(
            "Zentorch's INT8 kernels require the CPU to support AVX512 instructions."
            in str(context.exception)
        )

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
    def test_qlinear_incorrect_dtypes(
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

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim].to(
                    torch.bfloat16
                ),  # unsupported dtype
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
                self.data.y_scales[q_granularity_val],
                self.data.y_zero_points[q_granularity_val],
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported dtype for input tensor, only float32/uint8/int8 is supported"
            in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y,  # unsupported dtype
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
                self.data.y_scales[q_granularity_val],
                self.data.y_zero_points[q_granularity_val],
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported dtype for weight tensor, only int8 is supported"
            in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.input1d.to(torch.float16),  # unsupported dtype
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
                self.data.y_scales[q_granularity_val],
                self.data.y_zero_points[q_granularity_val],
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported dtype for bias tensor, only float32 is supported"
            in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"].to(torch.bfloat16),
                self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
                self.data.y_scales[q_granularity_val],
                self.data.y_zero_points[q_granularity_val],
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue("unsupported dtype for input_scales" in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"]["float32"][
                    q_zero_points_dtype
                ].to(torch.int32),
                self.data.y_scales[q_granularity_val],
                self.data.y_zero_points[q_granularity_val],
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported dtype for input_zero_points" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
                self.data.y_scales[q_granularity_val].to(torch.bfloat16),
                self.data.y_zero_points[q_granularity_val],
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue("unsupported dtype for weight_scales" in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
                self.data.y_scales[q_granularity_val],
                self.data.y_zero_points[q_granularity_val].to(torch.int32),
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported dtype for weight_zero_points" in str(context.exception)
        )

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
    def test_qlinear_incorrect_sizes(
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

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x1[0],  # incorrect size
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
                self.data.y_scales[q_granularity_val],
                self.data.y_zero_points[q_granularity_val],
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "shapes incompatible for matrix multiplication" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y1[0].to(torch.int8),  # incorrect size
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
                self.data.y_scales[q_granularity_val],
                self.data.y_zero_points[q_granularity_val],
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "shapes incompatible for matrix multiplication" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x3d,  # unsupported dims
                self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
                self.data.y_scales[q_granularity_val],
                self.data.y_zero_points[q_granularity_val],
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported dims for input_scales "
            "with respect to input tensor" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                self.data.y3d.to(torch.int8),  # unsupported dims
                self.data.y_scales[q_granularity_val],
                self.data.y_zero_points[q_granularity_val],
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported dims for input_zero_points "
            "with respect to input tensor" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
                self.data.y3d,  # unsupported dims
                self.data.y_zero_points[q_granularity_val],
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported dims for weight_scales "
            "with respect to weight tensor" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
                self.data.y_scales[q_granularity_val],
                self.data.y3d.to(torch.int8),  # unsupported dims
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported dims for weight_zero_points "
            "with respect to weight tensor" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.y_scales["per_channel"],  # per-channel scales not supported
                self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
                self.data.y_scales[q_granularity_val],
                self.data.y_zero_points[q_granularity_val],
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported number of elements for input_scales "
            "with respect to input tensor" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                self.data.y_zero_points["per_channel"],  # wrong num of zero points
                self.data.y_scales[q_granularity_val],
                self.data.y_zero_points[q_granularity_val],
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported number of elements for input_zero_points "
            "with respect to input tensor" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
                self.data.wrong_scales_per_channel,  # incorrect number of elements
                self.data.y_zero_points[q_granularity_val],
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported number of elements for weight_scales "
            "with respect to weight tensor" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
                self.data.y_scales[q_granularity_val],
                self.data.wrong_zero_points_per_channel,  # incorrect number of elements
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported number of elements for weight_zero_points "
            "with respect to weight tensor" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.input3d,  # unsupported dims for bias
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"]["float32"][q_zero_points_dtype],
                self.data.y_scales[q_granularity_val],
                self.data.y_zero_points[q_granularity_val],
                self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported dimensions for input/bias/self" in str(context.exception)
        )

    @parameterized.expand(
        product(
            qlinear_dtypes,
            input_dim_opt,
            q_weight_list_opt,
            bias_opt,
            q_granularity_opt,
            q_zero_points_dtype_opt,
            q_linear_dtype_opt,
        ),
        skip_on_empty=True,
    )
    @torch.inference_mode()
    def test_qlinear_output_dtype_support(
        self,
        dtype,
        input_dim,
        q_weight_idx,
        bias_opt_idx,
        q_granularity_val,
        q_zero_points_dtype,
        input_dtype,
    ):
        self.skip_if_bfloat16_not_yet_supported(dtype)
        self.data.create_data(dtype)
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear[input_dtype][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype],
                self.data.y_scales[q_granularity_val],
                self.data.y_zero_points[q_granularity_val],
                torch.float16,  # unsupported output dtype
            )
        self.assertTrue(
            "output_dtype received is not yet supported, only "
            "float32/uint8/int8 is supported" in str(context.exception)
        )

    @parameterized.expand(
        product(
            qlinear_dtypes,
            input_dim_opt,
            q_weight_list_opt,
            bias_opt,
            q_granularity_opt,
            q_zero_points_dtype_opt,
            q_linear_dtype_opt,
            q_linear_dtype_opt,
        ),
        skip_on_empty=True,
    )
    @torch.inference_mode()
    def test_qlinear_accuracy(
        self,
        dtype,
        input_dim,
        q_weight_idx,
        bias_opt_idx,
        q_granularity_val,
        q_zero_points_dtype,
        input_dtype,
        q_linear_dtype,
    ):
        self.skip_if_bfloat16_not_yet_supported(dtype)
        self.data.create_data(dtype)

        # simulated qlinear
        qdq_linear_output = qdq_linear(
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype],
            self.data.y_scales[q_granularity_val],
            self.data.y_zero_points[q_granularity_val],
            None,
            self.data.output_scales["per_tensor"][q_linear_dtype]["positive_scales"],
            self.data.output_zero_points["per_tensor"][q_linear_dtype],
        )
        # zentorch qlinear
        zentorch_qlinear_output = torch.ops.zentorch.zentorch_qlinear(
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype],
            self.data.y_scales[q_granularity_val],
            self.data.y_zero_points[q_granularity_val],
            (
                self.data.x_for_qlinear[q_linear_dtype][input_dim].dtype
                if q_linear_dtype == "float32"
                else self.data.output_zero_points["per_tensor"][q_linear_dtype].dtype
            ),
            self.data.output_scales["per_tensor"][q_linear_dtype]["positive_scales"],
            self.data.output_zero_points["per_tensor"][q_linear_dtype],
        )

        self.assertEqual(
            qdq_linear_output, zentorch_qlinear_output, atol=1e-3, rtol=1e-4
        )


if __name__ == "__main__":
    run_tests()
