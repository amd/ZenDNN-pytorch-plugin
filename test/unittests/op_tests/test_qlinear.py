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
    DataTypes,
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
    get_comp_zero_points,
)
from quant_utils import qdq_linear  # noqa: 402


@unittest.skipIf(not has_zentorch, "ZENTORCH is not installed")
class Test_Qlinear(QLinearTestCase):
    @QLinearTestCase.hypothesis_params_qlinear_itr(
        input_dim_opt_list=input_dim_opt,
        q_weight_list_opt_list=q_weight_list_opt,
        bias_opt_list=bias_opt,
        q_granularity_opt_list=q_granularity_opt,
        q_zero_points_dtype_opt_list=q_zero_points_dtype_opt,
        q_linear_dtype_opt_list=q_linear_dtype_opt,
        dtype_list=qlinear_dtypes,
    )
    @torch.inference_mode()
    def test_qlinear_unsupported_hardware(self):
        if zentorch._C.is_avx512_supported():
            self.skipTest(
                "Skipping hardware test, as AVX512 instructions are supported."
            )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][2],
                self.data.y_int8[0],
                self.data.bias_for_qlinear[0],
                self.data.x_scales["per_tensor"],
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"]["float32"]["uint8"]
                ),
                self.data.y_scales["per_channel"],
                get_comp_zero_points(self.data.y_zero_points["per_channel"]),
                output_dtype=self.data.x_for_qlinear["float32"][2].dtype,
            )
        self.assertTrue(
            "Zentorch's INT8 kernels require the CPU to support AVX512 instructions."
            in str(context.exception)
        )

    @QLinearTestCase.hypothesis_params_qlinear_itr(
        input_dim_opt_list=input_dim_opt,
        q_weight_list_opt_list=q_weight_list_opt,
        bias_opt_list=bias_opt,
        q_granularity_opt_list=q_granularity_opt,
        q_zero_points_dtype_opt_list=q_zero_points_dtype_opt,
        q_linear_dtype_opt_list=q_linear_dtype_opt,
        dtype_list=qlinear_dtypes,
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

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim].to(
                    torch.float16
                ),  # unsupported dtype
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"]["float32"][
                        q_zero_points_dtype
                    ]
                ),
                self.data.y_scales[q_granularity_val],
                get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
                output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported dtype for input tensor, only "
            "float32/bfloat16/uint8/int8 is supported" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y,  # unsupported dtype
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"]["float32"][
                        q_zero_points_dtype
                    ]
                ),
                self.data.y_scales[q_granularity_val],
                get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
                output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
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
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"]["float32"][
                        q_zero_points_dtype
                    ]
                ),
                self.data.y_scales[q_granularity_val],
                get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
                output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported dtype for bias tensor, only float32 or bfloat16 "
            "is supported" in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"].to(torch.float16),
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"]["float32"][
                        q_zero_points_dtype
                    ]
                ),
                self.data.y_scales[q_granularity_val],
                get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
                output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue("unsupported dtype for input_scales" in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                self.data.x_zero_points["per_tensor"]["float32"]["uint8"].to(
                    torch.float
                ),
                self.data.y_scales[q_granularity_val],
                get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
                output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
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
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"]["float32"][
                        q_zero_points_dtype
                    ]
                ),
                self.data.y_scales[q_granularity_val].to(torch.float16),
                get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
                output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue("unsupported dtype for weight_scales" in str(context.exception))

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"]["float32"][
                        q_zero_points_dtype
                    ]
                ),
                self.data.y_scales[q_granularity_val],
                self.data.y_zero_points[q_granularity_val].to(torch.float),
                output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported dtype for weight_zero_points" in str(context.exception)
        )

    @QLinearTestCase.hypothesis_params_qlinear_itr(
        input_dim_opt_list=input_dim_opt,
        q_weight_list_opt_list=q_weight_list_opt,
        bias_opt_list=bias_opt,
        q_granularity_opt_list=q_granularity_opt,
        q_zero_points_dtype_opt_list=q_zero_points_dtype_opt,
        q_linear_dtype_opt_list=q_linear_dtype_opt,
        dtype_list=qlinear_dtypes,
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

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x1[0],  # incorrect size
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"]["float32"][
                        q_zero_points_dtype
                    ]
                ),
                self.data.y_scales[q_granularity_val],
                get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
                output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
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
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"]["float32"][
                        q_zero_points_dtype
                    ]
                ),
                self.data.y_scales[q_granularity_val],
                get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
                output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
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
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"]["float32"][
                        q_zero_points_dtype
                    ]
                ),
                self.data.y_scales[q_granularity_val],
                get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
                output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
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
                get_comp_zero_points(self.data.y3d),  # unsupported dims
                self.data.y_scales[q_granularity_val],
                get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
                output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "only scalar and 1-d input_zero_points are supported"
            in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"]["float32"][
                        q_zero_points_dtype
                    ]
                ),
                self.data.y3d,  # unsupported dims
                get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
                output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
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
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"]["float32"][
                        q_zero_points_dtype
                    ]
                ),
                self.data.y_scales[q_granularity_val],
                get_comp_zero_points(self.data.y3d),  # unsupported dims
                output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "only scalar and 1-d weight_zero_points are supported"
            in str(context.exception)
        )

        # Skip qlinear test for n=1 because when self.data.y_scales["per_channel"].numel() == 1,
        # it refers to per-tensor input scales, which is a valid configuration, and thus we skip
        # this test for this valid scenario.
        if self.data.y_scales["per_channel"].numel() != 1:
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zentorch_qlinear(
                    self.data.x_for_qlinear["float32"][input_dim],
                    self.data.y_int8[q_weight_idx],
                    self.data.bias_for_qlinear[bias_opt_idx],
                    self.data.y_scales["per_channel"],  # per-channel scales not supported
                    get_comp_zero_points(
                        self.data.x_zero_points["per_tensor"]["float32"][
                            q_zero_points_dtype
                        ]
                    ),
                    self.data.y_scales[q_granularity_val],
                    get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
                    output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
                )
            self.assertTrue(
                "unsupported number of elements for input_scales "
                "with respect to input tensor" in str(context.exception)
            )

        # Skip qlinear test for n=1 because when self.data.y_zero_points["per_channel"].numel() == 1,
        # it refers to per-tensor input zero points, which is also a valid configuration, and therefore
        #  we skip this test for this valid scenario.
        if self.data.y_zero_points["per_channel"].numel() != 1:
            with self.assertRaises(RuntimeError) as context:
                torch.ops.zentorch.zentorch_qlinear(
                    self.data.x_for_qlinear["float32"][input_dim],
                    self.data.y_int8[q_weight_idx],
                    self.data.bias_for_qlinear[bias_opt_idx],
                    self.data.x_scales["per_tensor"],
                    self.data.y_zero_points["per_channel"].to(
                        torch.int32
                    ),  # wrong num of zero points
                    self.data.y_scales[q_granularity_val],
                    get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
                    output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
                )
            self.assertTrue(
                "only supporting per-tensor quantization for input"
                in str(context.exception)
            )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"]["float32"][
                        q_zero_points_dtype
                    ]
                ),
                self.data.wrong_scales_per_channel,  # incorrect number of elements
                get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
                output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
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
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"]["float32"][
                        q_zero_points_dtype
                    ]
                ),
                self.data.y_scales[q_granularity_val],
                self.data.wrong_zero_points_per_channel.to(
                    torch.int32
                ),  # incorrect number of elements
                output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "only supporting per-tensor and per-channel quantization for weight"
            in str(context.exception)
        )

        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear["float32"][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.input3d,  # unsupported dims for bias
                self.data.x_scales["per_tensor"],
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"]["float32"][
                        q_zero_points_dtype
                    ]
                ),
                self.data.y_scales[q_granularity_val],
                get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
                output_dtype=self.data.x_for_qlinear["float32"][input_dim].dtype,
            )
        self.assertTrue(
            "unsupported dimensions for input/bias/self" in str(context.exception)
        )

    @QLinearTestCase.hypothesis_params_qlinear_itr(
        input_dim_opt_list=input_dim_opt,
        q_weight_list_opt_list=q_weight_list_opt,
        bias_opt_list=bias_opt,
        q_granularity_opt_list=q_granularity_opt,
        q_zero_points_dtype_opt_list=q_zero_points_dtype_opt,
        q_linear_dtype_opt_list=q_linear_dtype_opt,
        dtype_list=qlinear_dtypes,
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
        with self.assertRaises(RuntimeError) as context:
            torch.ops.zentorch.zentorch_qlinear(
                self.data.x_for_qlinear[input_dtype][input_dim],
                self.data.y_int8[q_weight_idx],
                self.data.bias_for_qlinear[bias_opt_idx],
                self.data.x_scales["per_tensor"],
                get_comp_zero_points(
                    self.data.x_zero_points["per_tensor"][input_dtype][
                        q_zero_points_dtype
                    ]
                ),
                self.data.y_scales[q_granularity_val],
                get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
                output_dtype=torch.float16,  # unsupported output dtype
            )
        self.assertTrue(
            "output_dtype received is not yet supported, only "
            "float32/bfloat16/uint8/int8 is supported" in str(context.exception)
        )

    @QLinearTestCase.hypothesis_params_qlinear_itr(
        input_dim_opt_list=input_dim_opt,
        q_weight_list_opt_list=q_weight_list_opt,
        bias_opt_list=bias_opt,
        q_granularity_opt_list=q_granularity_opt,
        q_zero_points_dtype_opt_list=q_zero_points_dtype_opt,
        q_linear_dtype_opt_list=q_linear_dtype_opt,
        dtype_list=qlinear_dtypes,
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
        output_dtype,
    ):
        self.skip_if_does_not_support_arg_combination_for_qlinear(
            bias_opt_idx, input_dtype, output_dtype
        )
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
            DataTypes.get_torch_type(output_dtype),
            self.data.output_scales["per_tensor"][output_dtype]["positive_scales"],
            self.data.output_zero_points["per_tensor"][output_dtype],
        )
        # zentorch qlinear
        zentorch_qlinear_output = torch.ops.zentorch.zentorch_qlinear(
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            self.data.x_scales["per_tensor"],
            get_comp_zero_points(
                self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype]
            ),
            self.data.y_scales[q_granularity_val],
            get_comp_zero_points(self.data.y_zero_points[q_granularity_val]),
            output_dtype=DataTypes.get_torch_type(output_dtype),

            output_scales=self.data.output_scales["per_tensor"][output_dtype][
                "positive_scales"
            ],
            output_zero_points=get_comp_zero_points(
                self.data.output_zero_points["per_tensor"][output_dtype]
            ),
        )
        # bf16 qlinear comparsion requires slightly higher tolerance as compared to fp32
        self.assertEqual(
            qdq_linear_output, zentorch_qlinear_output, atol=1e-2, rtol=1e-2
        )

    @QLinearTestCase.hypothesis_params_qlinear_itr(
        input_dim_opt_list=input_dim_opt,
        q_weight_list_opt_list=q_weight_list_opt,
        bias_opt_list=bias_opt,
        q_granularity_opt_list=q_granularity_opt,
        q_zero_points_dtype_opt_list=q_zero_points_dtype_opt,
        q_linear_dtype_opt_list=q_linear_dtype_opt,
        dtype_list=qlinear_dtypes,
    )
    @torch.inference_mode()
    def test_qlinear_accuracy_0dim_scales(
        self,
        dtype,
        input_dim,
        q_weight_idx,
        bias_opt_idx,
        q_zero_points_dtype,
        input_dtype,
        output_dtype,
    ):
        self.skip_if_does_not_support_arg_combination_for_qlinear(
            bias_opt_idx, input_dtype, output_dtype
        )

        zero_dim_x_scales = torch.tensor(self.data.x_scales["per_tensor"].item())
        zero_dim_y_scales = torch.tensor(self.data.y_scales["per_tensor"].item())
        zero_dim_output_scales = (
            None
            if self.data.output_scales["per_tensor"][output_dtype]["positive_scales"]
            is None
            else torch.tensor(
                self.data.output_scales["per_tensor"][output_dtype][
                    "positive_scales"
                ].item()
            )
        )

        # simulated qlinear
        qdq_linear_output = qdq_linear(
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            zero_dim_x_scales,
            self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype],
            zero_dim_y_scales,
            self.data.y_zero_points["per_tensor"],
            None,
            DataTypes.get_torch_type(output_dtype),
            zero_dim_output_scales,
            self.data.output_zero_points["per_tensor"][output_dtype],
        )

        # zentorch qlinear
        zentorch_qlinear_output = torch.ops.zentorch.zentorch_qlinear(
            self.data.x_for_qlinear[input_dtype][input_dim],
            self.data.y_int8[q_weight_idx],
            self.data.bias_for_qlinear[bias_opt_idx],
            zero_dim_x_scales,
            get_comp_zero_points(
                self.data.x_zero_points["per_tensor"][input_dtype][q_zero_points_dtype]
            ),
            zero_dim_y_scales,
            get_comp_zero_points(self.data.y_zero_points["per_tensor"]),
            output_dtype=DataTypes.get_torch_type(output_dtype),
            output_scales=zero_dim_output_scales,
            output_zero_points=get_comp_zero_points(
                self.data.output_zero_points["per_tensor"][output_dtype]
            ),
        )

        self.assertEqual(
            qdq_linear_output, zentorch_qlinear_output, atol=1e-2, rtol=1e-2
        )


if __name__ == "__main__":
    run_tests()
